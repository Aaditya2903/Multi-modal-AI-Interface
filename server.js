require('dotenv').config();
const express = require('express');
const path = require('path');
const axios = require('axios');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const OpenAI = require('openai');
const { Civitai } = require('civitai'); // Import the SDK we just installed

const app = express();
const port = 3000;

// --- MIDDLEWARE ---
app.use(express.json());
app.use(express.static(path.join(__dirname)));

// --- API CONFIGURATION ---
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const geminiModel = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
const groq = new OpenAI({ baseURL: "https://api.groq.com/openai/v1", apiKey: process.env.GROQ_API_KEY });
const openRouter = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: process.env.OPENROUTER_API_KEY,
  defaultHeaders: { "HTTP-Referer": "http://localhost:3000", "X-Title": "Multi-Modal AI" }
});

// Initialize Civitai SDK
// Make sure CIVITAI_API_TOKEN is in your .env file!
const civitai = new Civitai({
    auth: process.env.CIVITAI_API_TOKEN
});

// --- SCORING ---
function calculateScore(text) {
    if (!text) return 0;
    const words = text.match(/\b(\w+)\b/g) || [];
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [];
    if (words.length === 0 || sentences.length === 0) return 0;
    const density = (text.match(/\b(?!is|the|and|a|to)\w+\b/gi) || []).length / words.length * 100;
    return Math.round((50 + density) - (words.length / sentences.length));
}

// --- HELPER: IMAGE PROMPT ---
async function createImagePrompt(userQuery) {
    try {
        const completion = await groq.chat.completions.create({
            model: "llama-3.1-8b-instant",
            messages: [{ "role": "user", "content": `Create a concise, visual image prompt for: "${userQuery}". Output ONLY the prompt.` }],
        });
        return completion.choices[0].message.content.replace(/"/g, '');
    } catch (e) { return userQuery; }
}

// --- CIVITAI GENERATION (SDK VERSION) ---
// --- CIVITAI GENERATION (SDK VERSION - FIXED) ---
// --- CIVITAI GENERATION (SDK VERSION - FIXED PARSING) ---
async function generateCivitaiImage(prompt, modelAir) {
    if (!process.env.CIVITAI_API_TOKEN) throw new Error("Civitai API Token missing");

    console.log(`[Civitai SDK] Generating for: "${prompt.substring(0, 15)}..." using ${modelAir}`);

    const input = {
        model: modelAir,
        params: {
            prompt: prompt,
            negativePrompt: "ugly, blurry, low quality",
            width: 1024, 
            height: 1024,
            steps: 25,
            cfgScale: 7
        }
    };

    try {
        const response = await civitai.image.fromText(input, true);
        
        let imageUrl = null;

        // PARSING LOGIC: Handle both Object and Array responses
        if (response.blobUrl) {
            imageUrl = response.blobUrl;
        } 
        else if (response.jobs && response.jobs.length > 0) {
            const job = response.jobs[0];
            
            // Check if 'result' is an Array (The issue you faced)
            if (Array.isArray(job.result) && job.result.length > 0) {
                imageUrl = job.result[0].blobUrl;
            } 
            // Check if 'result' is a single Object
            else if (job.result && job.result.blobUrl) {
                imageUrl = job.result.blobUrl;
            }
        }

        if (imageUrl) {
            console.log("[Civitai SDK] Success! Downloading image...");
            const imgRes = await axios.get(imageUrl, { responseType: 'arraybuffer' });
            return Buffer.from(imgRes.data, 'binary').toString('base64');
        } else {
            console.error("Civitai Response Dump:", JSON.stringify(response, null, 2));
            throw new Error("Job completed but no image URL found.");
        }

    } catch (error) {
        if (error.message.includes("Validation error")) {
             console.error("[Civitai Validation Error]", JSON.stringify(error.cause || error, null, 2));
        } else {
             console.error("[Civitai SDK Error]", error.message);
        }
        throw error; 
    }
}
// --- GENERATION ENDPOINT ---
app.post('/generate-all', async (req, res) => {
    try {
        const { query: userQuery, models, config } = req.body;
        if (!userQuery) return res.status(400).json({ error: 'Query required' });

        const cfg = {
            gemini: config?.gemini || "gemini-1.5-flash",
            groq: config?.groq || "llama-3.1-8b-instant",
            openrouter: config?.openrouter || "google/gemini-2.0-flash-exp:free",
            civitai: config?.civitai || "urn:air:sdxl:checkpoint:civitai:133005@348913" 
        };

        const askGemini = async () => {
            try {
                const model = genAI.getGenerativeModel({ model: cfg.gemini });
                const result = await model.generateContent(userQuery);
                return { model: `Gemini (${cfg.gemini})`, answer: result.response.text() };
            } catch (e) { return { model: 'Gemini', answer: `Error: ${e.message}`, error: true }; }
        };

        const askGroq = async () => {
            try {
                const completion = await groq.chat.completions.create({
                    model: cfg.groq, messages: [{ "role": "user", "content": userQuery }]
                });
                return { model: `Groq (${cfg.groq})`, answer: completion.choices[0].message.content };
            } catch (e) { return { model: 'Groq', answer: `Error: ${e.message}`, error: true }; }
        };

        const askOpenRouter = async () => {
            try {
                const completion = await openRouter.chat.completions.create({
                    model: cfg.openrouter, messages: [{ "role": "user", "content": userQuery }]
                });
                return { model: `OpenRouter (${cfg.openrouter})`, answer: completion.choices[0].message.content };
            } catch (e) { return { model: 'OpenRouter', answer: `Error: ${e.message}`, error: true }; }
        };

        const generateImageTask = async () => {
            let prompt = userQuery;
            try { prompt = await createImagePrompt(userQuery); } catch(e){}

            // 1. Try Civitai SDK
            if(models.image) { 
                try {
                    const base64 = await generateCivitaiImage(prompt, cfg.civitai);
                    return { type: 'image', data: base64 };
                } catch (civitaiError) {
                    // 2. Fallback to Pollinations (Anime mode forced as requested)
                    console.error("⚠️ Civitai Failed, using fallback...");
                    try {
                        const url = `https://image.pollinations.ai/prompt/${encodeURIComponent(prompt)}?width=1024&height=1024&nologo=true&model=anime`;
                        const response = await axios.get(url, { responseType: 'arraybuffer' });
                        return { type: 'image', data: Buffer.from(response.data).toString('base64') };
                    } catch (fallbackErr) {
                        return { type: 'image', data: null, error: true, message: "All image generators failed." };
                    }
                }
            }
            return { type: 'image', data: null };
        };

        const tasks = [];
        if (models.gemini) tasks.push(askGemini());
        if (models.groq) tasks.push(askGroq());
        if (models.openrouter) tasks.push(askOpenRouter());
        if (models.image) tasks.push(generateImageTask());

        const results = await Promise.all(tasks);
        
        const textResults = [];
        let imageResult = null;
        results.forEach(r => {
            if (r.type === 'image') imageResult = r.error ? null : r.data;
            else textResults.push({ ...r, score: r.error ? -999 : calculateScore(r.answer) });
        });

        res.json({ textResults, imageResult });

    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Server Error' });
    }
});

app.listen(port, () => {
    console.log(`🚀 Server running at http://localhost:${port}`);
});