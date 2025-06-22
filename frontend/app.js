// const backendUrl = "https://test-openai-chat.onrender.com"; // Replace with your actual backend
const backendUrl = "http:localhost:8000"; // Replace with your actual backend

const sendBtn = document.getElementById("send-btn");
const textInput = document.getElementById("text-input");
const status = document.getElementById("status");
const historyList = document.getElementById("history-list");
const clearHistoryBtn = document.getElementById("clear-history-btn");


let recognition;
let history = []; // { user, reply, timestamp, audio }


sendBtn.addEventListener("click", async () => {
  const message = textInput.value.trim();
  if (!message) return;
  textInput.value = "";
  status.textContent = "⏳ Sending text...";
  await processText(message);
});

async function processText(text) {
  try {
    const res = await fetch(`${backendUrl}/process_text/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await res.json();
    const timestamp = new Date().toLocaleTimeString();
    const entry = {
      user: text,
      reply: data.reply,
      timestamp,
    };

    history.push(entry);
    saveHistory();
    renderHistory();
    status.textContent = "🧠 GPT: " + data.reply;
  } catch (err) {
    console.error("Error processing:", err);
    status.textContent = "❌ Failed to process input.";
  }
}

function renderHistory() {
  historyList.innerHTML = "";

  history.forEach((entry, index) => {
    const li = document.createElement("li");
    li.className = "bg-gray-100 rounded p-2";

    li.innerHTML = `
      <div><strong>You:</strong> ${entry.user}</div>
      <div><strong>GPT:</strong> ${entry.reply}</div>
      <div class="text-xs text-gray-500">🕒 ${entry.timestamp}</div>
    `;

    historyList.appendChild(li);
  });


  historyList.scrollTop = historyList.scrollHeight;
}

function saveHistory() {
  localStorage.setItem("chatHistory", JSON.stringify(history));
}

function loadHistory() {
  const data = localStorage.getItem("chatHistory");
  if (data) {
    history = JSON.parse(data);
    renderHistory();
  }
}

clearHistoryBtn.addEventListener("click", () => {
  if (confirm("Are you sure you want to clear the chat history?")) {
    history = [];
    localStorage.removeItem("chatHistory");
    renderHistory();
    status.textContent = "🧹 Chat history cleared.";
  }
});

// Load on startup
loadHistory();