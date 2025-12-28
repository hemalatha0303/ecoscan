// script.js

function showPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(pageId).classList.add('active');
}

// Preview Image logic
document.getElementById('veg-input').onchange = e => preview(e, 'veg-preview');
document.getElementById('soil-input').onchange = e => preview(e, 'soil-preview');

function preview(event, id) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = () => {
            const img = document.getElementById(id);
            img.src = reader.result;
            img.style.display = 'block';
        }
        reader.readAsDataURL(file);
    }
}

async function processAI(type) {
    const input = type === 'vegetation' ? document.getElementById('veg-input') : document.getElementById('soil-input');
    const resultDiv = type === 'vegetation' ? document.getElementById('veg-result') : document.getElementById('soil-result');
    
    if (!input.files[0]) return alert("Please upload an image first!");

    resultDiv.innerHTML = "⏳ Processing with AI Model... Please wait.";
    resultDiv.style.color = "#121212"; // Reset to black while loading

    const formData = new FormData();
    formData.append('file', input.files[0]);

    // Use the local address where your backend terminal is running
    const API_BASE = "http://127.0.0.1:5000";

    try {
        const response = await fetch(`${API_BASE}/predict/${type}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error("Server responded with an error");
        }

        const data = await response.json();
        
        // Success UI updates
        resultDiv.style.color = "#2e7d32"; // Success Green
        if(type === 'vegetation') {
            resultDiv.innerHTML = `✅ Analysis Complete: <b>${data.coverage}%</b> Vegetation Detected`;
        } else {
            resultDiv.innerHTML = `✅ Detected Soil: <b>${data.label}</b> (${(data.confidence * 100).toFixed(2)}%)`;
        }

    } catch (error) {
        console.error("Error:", error);
        resultDiv.style.color = "red";
        resultDiv.innerHTML = "❌ Error connecting to AI Server. Make sure app.py is running.";
    }
}