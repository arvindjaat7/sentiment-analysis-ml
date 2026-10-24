async function analyzeSentiment() {
    const review = document.getElementById("review").value.trim();
    const result = document.getElementById("result");
    const sentiment = document.getElementById("sentiment");
    const confidence = document.getElementById("confidence");
    const error = document.getElementById("error");
    const button = document.getElementById("analyzeBtn");

    error.textContent = "";
    result.classList.add("hidden");

    if (!review) {
        error.textContent = "Please enter a review first.";
        return;
    }

    button.disabled = true;
    button.textContent = "Analyzing...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                review: review
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        sentiment.textContent =
            "Sentiment: " + data.sentiment;

        result.classList.toggle(
            "negative",
            data.sentiment.toLowerCase() === "negative"
        );

        confidence.textContent =
            data.confidence !== null
                ? "Model confidence: " + data.confidence + "%"
                : "";

        result.classList.remove("hidden");

    } catch (err) {
        error.textContent = err.message;
    } finally {
        button.disabled = false;
        button.textContent = "Analyze Sentiment";
    }
}