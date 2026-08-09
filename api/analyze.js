export const config = {
    maxDuration: 60
};

function getProvider() {
    if (process.env.FEATHERLESS_API_KEY) {
        return {
            name: "featherless",
            apiKey: process.env.FEATHERLESS_API_KEY,
            url: "https://api.featherless.ai/v1/chat/completions",
            model: "Qwen/Qwen2.5-7B-Instruct",
            supportsJsonMode: true
        };
    }
    return null;
}

async function callLLM(provider, prompt, signal) {
    const response = await fetch(provider.url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${provider.apiKey}`
        },
        signal,
        body: JSON.stringify({
            model: provider.model,
            messages: [{ role: "user", content: prompt }],
            ...(provider.supportsJsonMode ? { response_format: { type: "json_object" } } : {}),
            temperature: 0.01,
            max_tokens: 2200
        })
    });

    const rawResponseText = await response.text();
    let data;
    try {
        data = JSON.parse(rawResponseText);
    } catch (e) {
        throw new Error(`Provider Error: ${rawResponseText.substring(0, 100)}`);
    }

    if (!response.ok) throw new Error(data.error?.message || "API call failed.");
    if (!data.choices || !data.choices[0]?.message) throw new Error("Empty response.");

    const rawContent = data.choices[0].message.content;
    try {
        return JSON.parse(rawContent);
    } catch (e) {
        const cleaned = rawContent.replace(/```json/g, '').replace(/```/g, '').trim();
        return JSON.parse(cleaned);
    }
}

// Extract clean topic keywords by removing citation parentheticals and filler words
function buildTopicQuery(sentence) {
    let clean = sentence.replace(/\([^)]*\)/g, ' ').replace(/[A-Z][a-zA-Z]+ et al\.?/g, '');
    const stopwords = new Set([
        'about', 'which', 'their', 'there', 'these', 'those', 'because', 'however',
        'research', 'study', 'studies', 'indicates', 'demonstrates', 'according',
        'specifically', 'furthermore', 'moreover', 'suggests', 'found', 'reported'
    ]);
    const words = clean
        .replace(/[^a-zA-Z0-9\s]/g, ' ')
        .split(/\s+/)
        .filter(w => w.length > 4 && !stopwords.has(w.toLowerCase()));

    return words.slice(0, 4).join(' ') || "academic research";
}

// Extract "Author, YEAR" style citation info SEPARATELY so we can search for the
// exact cited work instead of only generic topic words. Without this, a real
// citation to a paper that just isn't in one database's index gets treated the
// same as a fabricated one, because we never actually searched for it.
function buildCitationSearchQuery(sentence) {
    const patterns = [
        /\(([A-Z][A-Za-z'\-]+(?:\s(?:&|and)\s[A-Z][A-Za-z'\-]+)?(?:\set\sal\.?)?),?\s*((?:19|20)\d{2})[a-z]?\)/,
        /([A-Z][A-Za-z'\-]+(?:\s(?:&|and)\s[A-Z][A-Za-z'\-]+)?\set\sal\.?)\s*\(?((?:19|20)\d{2})[a-z]?\)?/
    ];
    for (const p of patterns) {
        const m = sentence.match(p);
        if (m) return `${m[1].replace(/\set\sal\.?/i, '')} ${m[2]}`.trim();
    }
    return null;
}

function extractClaims(text) {
    const sentences = (text.match(/[^.!?]+[.!?]+/g) || [text]).map(s => s.trim());
    const claims = [];

    for (const sentence of sentences) {
        if (sentence.split(/\s+/).length < 5) continue;

        const hasStat = /\b\d{1,3}(\.\d+)?\s?%|\b\d{4,}\b/.test(sentence);
        const isCitation = /\([A-Za-z.\s&]+,\s*\d{4}\)|[A-Z][a-zA-Z]+ et al\./.test(sentence);

        let type = 'claim';
        if (hasStat) type = 'statistic';
        if (isCitation) type = 'citation';

        claims.push({
            type: type,
            textSegment: sentence,
            query: buildTopicQuery(sentence),
            citationQuery: isCitation ? buildCitationSearchQuery(sentence) : null
        });
    }
    // Limit to 4 claims per call to maintain high speed and avoid Vercel timeouts
    return claims.slice(0, 4);
}

async function searchSemanticScholar(query, signal) {
    if (!query || query.trim() === '') return [];
    try {
        const url = `https://api.semanticscholar.org/graph/v1/paper/search?query=${encodeURIComponent(query)}&limit=3&fields=title,authors,year,url`;
        const res = await fetch(url, { signal });
        if (!res.ok) return [];
        const data = await res.json();
        return (data.data || []).map(p => ({
            title: p.title,
            authors: p.authors ? p.authors.map(a => a.name).join(', ') : 'Unknown Author',
            year: p.year || 'N/A',
            url: p.url || 'N/A'
        }));
    } catch (e) {
        return [];
    }
}

async function searchCrossref(query, signal) {
    if (!query || query.trim() === '') return [];
    try {
        const url = `https://api.crossref.org/works?query.bibliographic=${encodeURIComponent(query)}&rows=3`;
        const res = await fetch(url, {
            signal,
            headers: { "User-Agent": "ProofPrint/1.0 (mailto:support@proofprint.app)" }
        });
        if (!res.ok) return [];
        const data = await res.json();
        const items = data.message?.items || [];
        return items.map(it => ({
            title: Array.isArray(it.title) ? (it.title[0] || 'Untitled') : (it.title || 'Untitled'),
            authors: (it.author || []).map(a => `${a.given || ''} ${a.family || ''}`.trim()).filter(Boolean).join(', ') || 'Unknown Author',
            year: it.issued?.['date-parts']?.[0]?.[0] || 'N/A',
            url: it.DOI ? `https://doi.org/${it.DOI}` : (it.URL || 'N/A')
        }));
    } catch (e) {
        return [];
    }
}

function dedupeEvidence(list) {
    const seen = new Set();
    const out = [];
    for (const p of list) {
        const key = (p.title || '').trim().toLowerCase();
        if (!key || seen.has(key)) continue;
        seen.add(key);
        out.push(p);
    }
    return out.slice(0, 5);
}

// Cache identical queries within a single request — chunks of the same document
// often produce overlapping topic keywords, so this avoids redundant round trips.
function makeCachedSearcher(fn, cache) {
    return async (query, signal) => {
        if (!query) return [];
        const key = `${fn.name}:${query}`;
        if (cache.has(key)) return cache.get(key);
        const result = await fn(query, signal);
        cache.set(key, result);
        return result;
    };
}

async function mapWithConcurrency(items, limit, fn) {
    const results = new Array(items.length);
    let i = 0;
    async function worker() {
        while (i < items.length) {
            const idx = i++;
            results[idx] = await fn(items[idx]);
        }
    }
    await Promise.all(new Array(Math.min(limit, items.length)).fill(0).map(worker));
    return results;
}

export default async function handler(req, res) {
    if (req.method !== 'POST') return res.status(405).json({ error: 'Use POST.' });

    const provider = getProvider();
    if (!provider) return res.status(500).json({ error: 'No API key set.' });

    const { text, citationStyle = "APA" } = req.body;
    if (!text) return res.status(400).json({ error: 'No text provided.' });

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 55000);

    try {
        const claims = extractClaims(text);
        const cache = new Map();
        const cachedSS = makeCachedSearcher(searchSemanticScholar, cache);
        const cachedCR = makeCachedSearcher(searchCrossref, cache);

        // Search BOTH Semantic Scholar (topic match) and CrossRef (exact author/year
        // match when the sentence looks like a citation) in parallel per claim, then
        // merge. Two independent databases catch far more real, valid citations than
        // one alone — a paper missing from one index is often present in the other.
        const evidenceLists = await mapWithConcurrency(claims, 4, async (claim) => {
            const [ssResults, crResults] = await Promise.all([
                cachedSS(claim.query, controller.signal),
                cachedCR(claim.citationQuery || claim.query, controller.signal)
            ]);
            return dedupeEvidence([...ssResults, ...crResults]);
        });

        // Claims where we found real evidence go to the LLM for judgment.
        // Claims where BOTH databases came back empty are marked "unverifiable"
        // directly, WITHOUT asking the LLM to guess — a search miss is not proof
        // a citation is fake, and it should never be presented as one.
        const withEvidence = [];
        const finalResults = new Array(claims.length);

        claims.forEach((c, i) => {
            const ev = evidenceLists[i];
            if (ev.length > 0) {
                withEvidence.push({ id: i + 1, claim: c, evidence: ev });
            } else {
                finalResults[i] = {
                    textSegment: c.textSegment,
                    type: c.type,
                    status: "unverifiable",
                    confidenceScore: 0,
                    evidenceType: "No Live Source Available",
                    riskLevel: "medium",
                    explanation: "No matching source was found in either database searched (Semantic Scholar, CrossRef). This does NOT mean the claim or citation is false — the source may be paywalled, a book, older, or indexed under a different title.",
                    improvement: "Manually verify this citation against the original source (publisher site, DOI, or library catalog), since no automated match was found.",
                    suggestedSource: "",
                    formattedCitation: "",
                    sourcesChecked: []
                };
            }
        });

        if (withEvidence.length === 0) {
            clearTimeout(timeoutId);
            return res.status(200).json({ results: finalResults });
        }

        const claimBlocks = withEvidence.map(({ id, claim: c, evidence: ev }) => {
            const evidenceText = ev.map(p => `    - Title: "${p.title}" | Authors: ${p.authors} | Year: ${p.year} | URL: ${p.url}`).join('\n');
            return `Claim ${id} [${c.type}]: "${c.textSegment}"\n  Live Evidence Found in Literature:\n${evidenceText}`;
        }).join('\n\n');

        const prompt = `
You are a zero-tolerance academic peer reviewer and strict citation auditor. Evaluate EACH claim against the "Live Evidence" provided under that specific claim.

${claimBlocks}

STRICT EVALUATION INSTRUCTIONS:
1. TRUE FACT WITH FAKE/BAD CITATION:
   - If the factual assertion in the text is true or plausible, but the named author or year in the text does NOT explicitly exist in the Live Evidence: mark status as "unverified".
   - In "explanation": State that the background fact is valid/accurate, but explicitly call out that the cited author/year is fabricated or incorrect.
   - In "suggestedSource" & "formattedCitation": Pick a REAL paper from the Live Evidence list that supports this factual claim, and format it properly in **${citationStyle}** style.

2. UNTRUE / FALSE FACT:
   - If the claim itself is factually false, inaccurate, or absurd: mark status as "hallucination".
   - In "explanation": Explain clearly why the statement is factually incorrect.
   - In "suggestedSource" & "formattedCitation": Select a REAL paper from the Live Evidence list that provides real-world context on this topic, and format it properly in **${citationStyle}** style.

3. FORBIDDEN GENERIC TEXT: You are STRICTLY PROHIBITED from giving generic advice such as "conduct thorough research", "search Google Scholar", or "verify with a credible source".

4. MANDATORY REAL SOURCE & CITATION: Every item marked as "unverified" or "hallucination" MUST have:
   - "suggestedSource": The exact Title, Authors, Year, and URL of one of the real studies listed in the Live Evidence block.
   - "formattedCitation": A full, ready-to-copy citation of that exact real study formatted strictly in **${citationStyle}** style.

Return EXACTLY this JSON shape:
{
  "results": [
    {
      "id": 1,
      "textSegment": "exact original text segment",
      "type": "citation" | "statistic" | "quote" | "claim",
      "status": "verified" | "hallucination" | "unverified" | "format_error",
      "confidenceScore": 99,
      "evidenceType": "Peer-Reviewed Study" | "No Live Source Available",
      "riskLevel": "low" | "medium" | "high",
      "explanation": "2-sentence audit detailing factual accuracy and citation authenticity.",
      "improvement": "Actionable instructions replacing the fake/flawed source with the real study provided.",
      "suggestedSource": "Exact Title, Authors, Year, and URL of the real study from the Live Evidence.",
      "formattedCitation": "Full citation formatted strictly in ${citationStyle} style."
    }
  ]
}`;

        const resultObj = await callLLM(provider, prompt, controller.signal);
        clearTimeout(timeoutId);

        // Map each claim id back to its full raw evidence list, so we can show the
        // user everything that was actually checked — not just the one source the
        // model chose to cite in "suggestedSource".
        const evidenceById = new Map(withEvidence.map(({ id, evidence }) => [id, evidence]));

        const llmResults = resultObj.results || [];
        llmResults.forEach(r => {
            const idx = (r.id || 0) - 1;
            if (idx >= 0 && idx < finalResults.length) {
                r.sourcesChecked = evidenceById.get(r.id) || [];
                finalResults[idx] = r;
            }
        });
        // Safety net: if the model dropped an id or mismatched count, fill any
        // remaining gaps in original order so no claim silently disappears.
        let fallbackPtr = 0;
        for (let i = 0; i < finalResults.length; i++) {
            if (!finalResults[i] && fallbackPtr < llmResults.length) {
                const r = llmResults[fallbackPtr++];
                r.sourcesChecked = r.sourcesChecked || [];
                finalResults[i] = r;
            }
        }

        return res.status(200).json({ results: finalResults.filter(Boolean) });

    } catch (error) {
        clearTimeout(timeoutId);
        if (error.name === 'AbortError') return res.status(504).json({ error: "Timeout." });
        return res.status(500).json({ error: error.message });
    }
}