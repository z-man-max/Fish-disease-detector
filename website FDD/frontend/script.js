// DOM elements
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultDiv = document.getElementById("result");
const loadingDiv = document.getElementById("loading");
const backendStatus = document.getElementById("backendStatus");

// Backend URL
const BACKEND_URL = "http://192.168.1.10:5000";

// Check backend on load
checkBackend();

// When user selects image
fileInput.addEventListener("change", function (e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            preview.src = e.target.result;
            preview.style.display = "block";
            analyzeBtn.disabled = false;
            resultDiv.style.display = "none";
        };
        reader.readAsDataURL(file);
    }
});

// Check if backend is running
async function checkBackend() {
    try {
        const response = await fetch(`${BACKEND_URL}/api/health`);
        if (response.ok) {
            backendStatus.innerHTML = "✅ Connected";
            backendStatus.style.color = "green";
        } else {
            backendStatus.innerHTML = "❌ Not connected";
            backendStatus.style.color = "red";
        }
    } catch (error) {
        backendStatus.innerHTML = "❌ Cannot reach backend";
        backendStatus.style.color = "red";
        console.error("Backend error:", error);
    }
}

// Main function: Send image to backend
async function analyzeFish() {
    if (!fileInput.files[0]) {
        alert("Please select an image first!");
        return;
    }

    // Show loading
    loadingDiv.style.display = "block";
    resultDiv.style.display = "none";

    try {
        // Create FormData to send file
        const formData = new FormData();
        formData.append("image", fileInput.files[0]);

        console.log("📤 Sending image to backend...");

        // Send to Flask backend
        const response = await fetch(`${BACKEND_URL}/api/analyze`, {
            method: "POST",
            body: formData,
            // Note: Don't set Content-Type for FormData!
        });

        if (!response.ok) {
            throw new Error(`Backend error: ${response.status}`);
        }

        const data = await response.json();
        console.log("✅ Backend response:", data);

        // Display results
        displayResults(data);
    } catch (error) {
        console.error("❌ Error:", error);
        resultDiv.innerHTML = `
            <h3>Error</h3>
            <p>Failed to analyze image: ${error.message}</p>
            <p>Make sure Flask backend is running!</p>
        `;
        resultDiv.style.display = "block";
    } finally {
        loadingDiv.style.display = "none";
    }
}

// Display results from backend
function displayResults(data) {
    if (data.error) {
        resultDiv.innerHTML = `<h3>Error</h3><p>${data.error}</p>`;
    } else {
        const analysis = data.analysis;
        const statusClass = analysis.status === "infected" ? "infected" : "healthy";

        resultDiv.innerHTML = `
            <h3>${analysis.diagnosis}</h3>
            <div class="status ${statusClass}">
                Confidence: ${analysis.confidence}%
            </div>
            
            <h4>Detected Diseases:</h4>
            <ul>
                ${analysis.diseases_detected.map(disease => `
                    <li><strong>${disease.name}</strong> (${disease.confidence}%) - ${disease.description}</li>
                `).join("")}
            </ul>
            
            <h4>Recommendations:</h4>
            <ol>
                ${analysis.recommendations.map(rec => `<li>${rec}</li>`).join("")}
            </ol>
            
            <p><em>${analysis.note}</em></p>
            
            <hr>
            <p><small>Image: ${data.image_info.filename}</small></p>
        `;
    }

    resultDiv.style.display = "block";
    resultDiv.scrollIntoView({ behavior: "smooth" });
}