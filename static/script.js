
async function analyzeSentiment() {
    const review = document.getElementById("review").value.trim();
    const result = document.getElementById("result");
    const sentiment = document.getElementById("sentiment");
    const error = document.getElementById("error");
    const button = document.getElementById("analyzeBtn");

    error.textContent = "";
    result.classList.add("hidden");

    if (!review) {
        error.textContent = "Please enter some text first.";
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
            "Predicted Sentiment: " + data.sentiment.toUpperCase();

        document.getElementById("positivePercent").textContent =
            (data.percentages.positive ?? 0) + "%";

        document.getElementById("negativePercent").textContent =
            (data.percentages.negative ?? 0) + "%";

        document.getElementById("neutralPercent").textContent =
            (data.percentages.neutral ?? 0) + "%";

        result.classList.toggle(
            "negative",
            data.sentiment.toLowerCase() === "negative"
        );

        result.classList.remove("hidden");

    } catch (err) {
        error.textContent = err.message;
    } finally {
        button.disabled = false;
        button.textContent = "Analyze Sentiment";
    }
}
