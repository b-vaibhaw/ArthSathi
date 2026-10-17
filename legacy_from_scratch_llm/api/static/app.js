// app.js - Full Stack Frontend Logic for ArthaSathi

const API_BASE = window.location.origin;
let USER_ID = localStorage.getItem("arthasathi_user_id") || "";
let USERNAME = localStorage.getItem("arthasathi_username") || "";
let authMode = "login"; // "login" or "register"

// App State
let appState = {
    income: 25000,
    debts: [],
    transactions: [],
    language: "hi"
};

// DOM Elements
document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

function initApp() {
    checkHealth();
    setupEventListeners();
    
    // Theme Initialisation
    initTheme();
    
    // Auth Check
    if (checkAuth()) {
        loadUserProfile();
    }
}

// Check system health
async function checkHealth() {
    const badge = document.getElementById("api-status-badge");
    const text = document.getElementById("api-status-text");
    
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        
        if (data.status === "healthy") {
            badge.className = "status-badge online";
            text.textContent = "All Systems Online";
        } else {
            badge.className = "status-badge checking";
            text.textContent = "Server Starting...";
        }
    } catch (e) {
        badge.className = "status-badge offline";
        text.textContent = "Server Offline (Using local simulation)";
        console.warn("FastAPI server offline, falling back to local simulation.", e);
    }
}

// Load user profile from DB
async function loadUserProfile() {
    try {
        const res = await fetch(`${API_BASE}/user/${USER_ID}/profile`);
        if (res.ok) {
            const data = await res.json();
            appState.income = data.monthly_income || 25000;
            appState.debts = data.debts || [];
            appState.transactions = data.transactions || [];
            
            updateProfileUI();
            renderTransactions();
        }
    } catch (e) {
        // Fallback to defaults
        updateProfileUI();
        renderTransactions();
    }
}

// Update Profile & Summary UI elements
function updateProfileUI() {
    document.getElementById("profile-income-input").value = appState.income;
    
    const totalDebt = appState.debts.reduce((sum, d) => sum + d.principal, 0);
    document.getElementById("profile-total-debt").textContent = totalDebt.toLocaleString("en-IN");
    
    const totalEmi = appState.debts.reduce((sum, d) => sum + d.min_payment, 0);
    const dti = appState.income > 0 ? Math.round((totalEmi / appState.income) * 100) : 0;
    
    const dtiEl = document.getElementById("profile-dti");
    dtiEl.textContent = dti;
    
    const dtiCard = dtiEl.parentElement;
    if (dti > 40) {
        dtiCard.className = "metric-value-wrapper text-danger";
    } else if (dti > 25) {
        dtiCard.className = "metric-value-wrapper text-warning";
    } else {
        dtiCard.className = "metric-value-wrapper text-success";
    }
    
    renderDebtsTable();
    calculateDebtPayoff();
}

// Render debts in the table
function renderDebtsTable() {
    const tableBody = document.querySelector("#debts-list-table tbody");
    tableBody.innerHTML = "";
    
    if (appState.debts.length === 0) {
        tableBody.innerHTML = `
            <tr class="empty-debt-row">
                <td colspan="6">No debts registered. Add one using the form.</td>
            </tr>
        `;
        return;
    }
    
    appState.debts.forEach((debt, index) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><strong>${debt.name}</strong></td>
            <td>₹${debt.principal.toLocaleString("en-IN")}</td>
            <td>${debt.annual_rate}%</td>
            <td>₹${debt.min_payment.toLocaleString("en-IN")}</td>
            <td><span class="text-secondary" style="text-transform: capitalize;">${debt.lender_type}</span></td>
            <td><button class="btn-delete" data-index="${index}"><i class="fa-solid fa-trash"></i></button></td>
        `;
        tableBody.appendChild(tr);
    });
    
    // Add delete events
    document.querySelectorAll(".btn-delete").forEach(btn => {
        btn.addEventListener("click", (e) => {
            const index = e.currentTarget.getAttribute("data-index");
            appState.debts.splice(index, 1);
            updateProfileUI();
        });
    });
}

// Calculate Debt Payoff Strategy
function calculateDebtPayoff() {
    const section = document.getElementById("payoff-results-section");
    if (appState.debts.length === 0) {
        section.classList.add("hidden");
        return;
    }
    
    section.classList.remove("hidden");
    
    // Perform Avalanche vs Snowball math
    // Avalanche: sort by interest rate descending
    const avalancheOrder = [...appState.debts].sort((a, b) => b.annual_rate - a.annual_rate);
    // Snowball: sort by principal ascending
    const snowballOrder = [...appState.debts].sort((a, b) => a.principal - b.principal);
    
    // Render list
    const avList = document.getElementById("avalanche-list");
    avList.innerHTML = "";
    avalancheOrder.forEach((d, i) => {
        const li = document.createElement("li");
        li.innerHTML = `
            <div>
                <span class="item-name">#${i+1} ${d.name}</span>
                <span class="item-details">₹${d.principal.toLocaleString("en-IN")} @ ${d.annual_rate}%</span>
            </div>
            <div class="item-meta">
                <span class="item-pay">Pay ₹${d.min_payment.toLocaleString("en-IN")}</span>
                <div class="item-months">Highest rate priority</div>
            </div>
        `;
        avList.appendChild(li);
    });
    
    const sbList = document.getElementById("snowball-list");
    sbList.innerHTML = "";
    snowballOrder.forEach((d, i) => {
        const li = document.createElement("li");
        li.innerHTML = `
            <div>
                <span class="item-name">#${i+1} ${d.name}</span>
                <span class="item-details">₹${d.principal.toLocaleString("en-IN")} @ ${d.annual_rate}%</span>
            </div>
            <div class="item-meta">
                <span class="item-pay">Pay ₹${d.min_payment.toLocaleString("en-IN")}</span>
                <div class="item-months">Smallest balance priority</div>
            </div>
        `;
        sbList.appendChild(li);
    });
    
    // Estimate savings (avalanche vs snowball interest savings placeholder)
    let interestSaved = 0;
    appState.debts.forEach(d => {
        // rough savings estimate over credit card standard interest rates
        if (d.annual_rate > 24) {
            interestSaved += (d.principal * (d.annual_rate - 12) / 100) * 1.5; // roughly 1.5 years savings
        }
    });
    
    document.getElementById("interest-savings-val").textContent = `₹${Math.round(interestSaved).toLocaleString("en-IN")}`;
}

// Setup listeners
function setupEventListeners() {
    // Setup Auth and Theme listeners
    setupAuthAndThemeListeners();

    // Tab Switching
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", (e) => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
            
            e.currentTarget.classList.add("active");
            const tabId = e.currentTarget.getAttribute("data-tab");
            document.getElementById(tabId).classList.add("active");
        });
    });
    
    // Language change
    document.getElementById("user-language").addEventListener("change", (e) => {
        appState.language = e.target.value;
    });
    
    // Income update
    document.getElementById("profile-income-input").addEventListener("change", async (e) => {
        appState.income = parseFloat(e.target.value) || 0;
        updateProfileUI();
        
        // Sync with API
        try {
            await fetch(`${API_BASE}/user/income`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ user_id: USER_ID, monthly_income: appState.income })
            });
        } catch (err) {
            console.warn("Failed to sync income to server", err);
        }
    });
    
    // Chat Submit
    document.getElementById("chat-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const input = document.getElementById("chat-input");
        const query = input.value.trim();
        if (!query) return;
        
        input.value = "";
        await sendMessage(query);
    });
    
    // Suggestion buttons
    document.querySelectorAll(".suggest-btn").forEach(btn => {
        btn.addEventListener("click", async (e) => {
            const query = e.target.getAttribute("data-query");
            await sendMessage(query);
        });
    });
    
    // Add Loan
    document.getElementById("add-debt-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const name = document.getElementById("debt-name").value;
        const principal = parseFloat(document.getElementById("debt-principal").value);
        const rate = parseFloat(document.getElementById("debt-rate").value);
        const emi = parseFloat(document.getElementById("debt-emi").value);
        const type = document.getElementById("debt-lender-type").value;
        
        const newDebt = { name, principal, annual_rate: rate, min_payment: emi, lender_type: type };
        appState.debts.push(newDebt);
        updateProfileUI();
        
        // Clear form
        document.getElementById("add-debt-form").reset();
        
        // Sync with API
        try {
            await fetch(`${API_BASE}/user/debt`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ user_id: USER_ID, ...newDebt })
            });
        } catch (err) {
            console.warn("Failed to sync debt to server", err);
        }
    });
    
    // Modal Transactions
    const txnDialog = document.getElementById("txn-dialog");
    document.getElementById("add-txn-btn").addEventListener("click", () => {
        txnDialog.classList.remove("hidden");
    });
    document.getElementById("modal-txn-cancel").addEventListener("click", () => {
        txnDialog.classList.add("hidden");
    });
    document.getElementById("modal-txn-save").addEventListener("click", async () => {
        const type = document.getElementById("modal-txn-type").value;
        const amount = parseFloat(document.getElementById("modal-txn-amount").value);
        const category = document.getElementById("modal-txn-cat").value;
        const description = document.getElementById("modal-txn-desc").value;
        
        const newTxn = {
            date: new Date().toLocaleDateString("en-IN"),
            type,
            amount,
            category,
            description
        };
        
        appState.transactions.unshift(newTxn);
        renderTransactions();
        txnDialog.classList.add("hidden");
        
        // Sync with API
        try {
            await fetch(`${API_BASE}/user/transaction`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ user_id: USER_ID, type, amount, category, description })
            });
        } catch (err) {
            console.warn("Failed to sync txn to server", err);
        }
    });
    
    // Business Pricing Calc
    document.getElementById("calc-price-btn").addEventListener("click", () => {
        const cog = parseFloat(document.getElementById("pricing-cog").value) || 0;
        const opex = parseFloat(document.getElementById("pricing-opex").value) || 0;
        const margin = parseFloat(document.getElementById("pricing-margin").value) || 20;
        const competition = document.getElementById("pricing-competition").value;
        
        const totalCost = cog + opex;
        const multipliers = { low: 1.35, medium: 1.22, high: 1.12 };
        const mult = multipliers[competition];
        
        const suggested = totalCost * mult;
        const target = totalCost * (1 + margin / 100);
        const breakeven = totalCost * 1.05;
        
        document.getElementById("pricing-res-total-cost").textContent = `₹${totalCost.toFixed(2)}`;
        document.getElementById("pricing-res-suggested").textContent = `₹${suggested.toFixed(2)}`;
        document.getElementById("pricing-res-target").textContent = `₹${target.toFixed(2)}`;
        document.getElementById("pricing-res-min").textContent = `₹${breakeven.toFixed(2)}`;
        
        const competitionTips = {
            low: "Spacious margin. You have pricing power. Offer bundles to sell more.",
            medium: "Fair competitive rate. Keep watch on nearby street vendors.",
            high: "Tight competition! Stand out through clean packing or quick service."
        };
        document.getElementById("pricing-res-note").textContent = competitionTips[competition];
        
        document.getElementById("pricing-results").classList.remove("hidden");
    });
    
    // GST Checker
    document.getElementById("calc-gst-btn").addEventListener("click", () => {
        const turnover = parseFloat(document.getElementById("gst-turnover").value) || 0;
        const state = document.getElementById("gst-state").value;
        const type = document.getElementById("gst-type").value;
        
        let mandatoryLimit = type === "services" ? 2000000 : 4000000;
        let compositionLimit = type === "services" ? 1500000 : 3000000;
        
        if (state === "special") {
            mandatoryLimit = 1000000;
            compositionLimit = 750000;
        }
        
        let status = "Not Required";
        let scheme = "Exempt Scheme";
        let advice = "Your annual turnover is below the registration limits. You do not need to register. Avoid compliance paperwork!";
        let step = "No action needed. Keep recording your transactions locally in ArthaSathi.";
        
        if (turnover >= mandatoryLimit) {
            status = "Mandatory Registration";
            scheme = "Regular GST Scheme (18%)";
            advice = "Your turnover exceeds the legal exemption threshold. You are required by law to register for GST.";
            step = "Visit the GST portal (gst.gov.in) or your nearest Jan Sewa Kendra to apply with Aadhaar/PAN.";
        } else if (turnover >= compositionLimit) {
            status = "Optional composition";
            scheme = "Composition Scheme (1% GST)";
            advice = "You qualify for the lower taxation Composition Scheme. Pay only 1% flat rate on sales with quarterly returns.";
            step = "Apply for GST registration and elect the Composition Scheme option during registration.";
        }
        
        document.getElementById("gst-res-status").textContent = status;
        document.getElementById("gst-res-scheme").textContent = scheme;
        document.getElementById("gst-res-advice").textContent = advice;
        document.getElementById("gst-res-step").textContent = step;
        
        document.getElementById("gst-results").classList.remove("hidden");
    });
    
    // Income Tax Checker
    document.getElementById("calc-tax-btn").addEventListener("click", () => {
        const income = parseFloat(document.getElementById("tax-income").value) || 0;
        const age = parseInt(document.getElementById("tax-age").value) || 35;
        const deductions = parseFloat(document.getElementById("tax-deductions").value) || 0;
        
        // Simple Slab Calculations
        // New Regime (FY 2023-24)
        let newTax = 0;
        if (income <= 700000) {
            newTax = 0; // Section 87A rebate
        } else {
            // Slabs: 0-3L (0%), 3-6L (5%), 6-9L (10%), 9-12L (15%), 12-15L (20%), 15L+ (30%)
            let remaining = income;
            if (remaining > 1500000) { newTax += (remaining - 1500000) * 0.30; remaining = 1500000; }
            if (remaining > 1200000) { newTax += (remaining - 1200000) * 0.20; remaining = 1200000; }
            if (remaining > 900000)  { newTax += (remaining - 900000) * 0.15;  remaining = 900000; }
            if (remaining > 600000)  { newTax += (remaining - 600000) * 0.10;  remaining = 600000; }
            if (remaining > 300000)  { newTax += (remaining - 300000) * 0.05;  remaining = 300000; }
        }
        let newCess = newTax * 0.04;
        let finalNewTax = newTax + newCess;
        
        // Old Regime
        let oldTaxable = Math.max(0, income - deductions);
        let oldTax = 0;
        if (oldTaxable <= 500000) {
            oldTax = 0; // Rebate
        } else {
            // Slabs: 0-2.5L (0%), 2.5-5L (5%), 5-10L (20%), 10L+ (30%)
            let remaining = oldTaxable;
            if (remaining > 1000000) { oldTax += (remaining - 1000000) * 0.30; remaining = 1000000; }
            if (remaining > 500000)  { oldTax += (remaining - 500000) * 0.20;  remaining = 500000; }
            if (remaining > 250000)  { oldTax += (remaining - 250000) * 0.05;  remaining = 250000; }
        }
        let oldCess = oldTax * 0.04;
        let finalOldTax = oldTax + oldCess;
        
        document.getElementById("tax-res-new-val").textContent = `₹${Math.round(finalNewTax).toLocaleString("en-IN")}`;
        document.getElementById("tax-res-new-rate").textContent = `${((finalNewTax / income) * 100).toFixed(1)}% effective`;
        
        document.getElementById("tax-res-old-val").textContent = `₹${Math.round(finalOldTax).toLocaleString("en-IN")}`;
        document.getElementById("tax-res-old-rate").textContent = `${((finalOldTax / income) * 100).toFixed(1)}% effective`;
        
        const recommend = finalNewTax < finalOldTax ? "New Regime is cheaper for you!" : "Old Regime is cheaper based on your deductions!";
        document.getElementById("tax-res-recommendation").textContent = recommend;
        
        document.getElementById("tax-results").classList.remove("hidden");
    });
    
    // Credit readiness profiler
    document.getElementById("calc-credit-btn").addEventListener("click", () => {
        const status = appState.income > 15000 ? "High Creditworthiness" : "Basic/Medium Creditworthiness";
        const maxLoan = appState.income * 6;
        const emi = Math.round(maxLoan / 12);
        
        document.getElementById("credit-res-status").textContent = status;
        document.getElementById("credit-res-loan").textContent = `₹${maxLoan.toLocaleString("en-IN")}`;
        document.getElementById("credit-res-emi").textContent = `₹${emi.toLocaleString("en-IN")} (12 Months)`;
        document.getElementById("credit-res-regularity").textContent = "Stable income flow";
        
        const lendersList = document.getElementById("credit-res-lenders");
        lendersList.innerHTML = `
            <li><strong>MUDRA Shishu Scheme</strong>: Collateral-free govt loans up to ₹50,000 (apply at SBI or Bank of Baroda).</li>
            <li><strong>Regulated NBFC-MFI</strong>: Ujjivan or Annapurna Microfinance (capped at 24% annual interest).</li>
        `;
        
        const docsList = document.getElementById("credit-res-docs");
        docsList.innerHTML = `
            <li>Aadhaar card + PAN card</li>
            <li>Last 3 months of bank passbook statement</li>
            <li>2 passport photos + shop outline picture</li>
        `;
        
        document.getElementById("credit-results").classList.remove("hidden");
    });
    
    // Voice simulation modals
    const voiceSimBtn = document.getElementById("voice-simulate-btn");
    const voiceDialog = document.getElementById("voice-dialog");
    const voiceClose = document.getElementById("voice-dialog-close");
    
    voiceSimBtn.addEventListener("click", () => {
        voiceDialog.classList.remove("hidden");
    });
    
    voiceClose.addEventListener("click", () => {
        voiceDialog.classList.add("hidden");
    });
    
    document.querySelectorAll(".voice-opt-btn").forEach(btn => {
        btn.addEventListener("click", async (e) => {
            const text = e.currentTarget.getAttribute("data-audio-text");
            voiceDialog.classList.add("hidden");
            await sendMessage(text, true);
        });
    });
}

// Render transactions table
function renderTransactions() {
    const tbody = document.querySelector("#transactions-table tbody");
    tbody.innerHTML = "";
    
    if (appState.transactions.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-row">
                <td colspan="5">No transactions recorded. Talk to ArthaSathi to record via voice note!</td>
            </tr>
        `;
        return;
    }
    
    appState.transactions.forEach(t => {
        const tr = document.createElement("tr");
        const badgeClass = t.type === "income" ? "text-success" : "text-danger";
        tr.innerHTML = `
            <td>${t.date}</td>
            <td><strong class="${badgeClass}">${t.type.toUpperCase()}</strong></td>
            <td>₹${t.amount.toLocaleString("en-IN")}</td>
            <td>${t.category}</td>
            <td>${t.description}</td>
        `;
        tbody.appendChild(tr);
    });
}

// Send chat message
async function sendMessage(text, isVoice = false) {
    appendMessage(text, "user", isVoice);
    
    // Show typing loader
    const loaderId = appendTypingIndicator();
    
    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                user_id: USER_ID,
                message: text,
                language: appState.language
            })
        });
        
        removeTypingIndicator(loaderId);
        
        if (res.ok) {
            const data = await res.json();
            appendMessage(data.response, "assistant");
            
            // Check if transaction was parsed
            checkForTransactions(text);
        } else {
            // Template fallback if server fails
            appendMessage(getTemplateResponse(text), "assistant");
        }
    } catch (e) {
        removeTypingIndicator(loaderId);
        appendMessage(getTemplateResponse(text), "assistant");
    }
}

// Check for transactions matching regex locally
function checkForTransactions(text) {
    // Basic heuristics to match voice note additions
    const textLower = text.toLowerCase();
    let isIncome = textLower.includes("bicha") || textLower.includes("mila") || textLower.includes("earn");
    let isExpense = textLower.includes("kharcha") || textLower.includes("spent") || textLower.includes("petrol");
    
    const numMatch = text.replace(/,/g, '').match(/\b\d+\b/);
    if (numMatch && (isIncome || isExpense)) {
        const amt = parseFloat(numMatch[0]);
        let cat = "business";
        if (textLower.includes("petrol")) cat = "transport";
        if (textLower.includes("kiraya") || textLower.includes("rent")) cat = "rent";
        
        const t = {
            date: new Date().toLocaleDateString("en-IN"),
            type: isIncome ? "income" : "expense",
            amount: amt,
            category: cat,
            description: text
        };
        appState.transactions.unshift(t);
        renderTransactions();
    }
}

// Append chat message to chatbox
function appendMessage(text, sender, isVoice = false) {
    const container = document.getElementById("chat-messages");
    const msg = document.createElement("div");
    msg.className = `message ${sender}`;
    
    const avatarIcon = sender === "user" ? "fa-user" : "fa-robot";
    const voiceIcon = isVoice ? '<i class="fa-solid fa-microphone text-danger" style="margin-right: 6px;" title="Voice transcription"></i>' : "";
    
    msg.innerHTML = `
        <div class="message-avatar"><i class="fa-solid ${avatarIcon}"></i></div>
        <div class="message-content">
            <p>${voiceIcon}${text}</p>
            <span class="message-time">${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
        </div>
    `;
    
    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
}

// Typing loader
function appendTypingIndicator() {
    const container = document.getElementById("chat-messages");
    const loader = document.createElement("div");
    const id = "loader_" + Date.now();
    loader.id = id;
    loader.className = "message assistant";
    loader.innerHTML = `
        <div class="message-avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="message-content" style="padding: 10px 16px;">
            <div class="voice-wave" style="height: 16px; margin: 0; gap: 3px;">
                <span class="stroke" style="width: 3px; height: 6px; background: var(--primary);"></span>
                <span class="stroke" style="width: 3px; height: 6px; background: var(--primary); animation-delay: 0.2s;"></span>
                <span class="stroke" style="width: 3px; height: 6px; background: var(--primary); animation-delay: 0.4s;"></span>
            </div>
        </div>
    `;
    container.appendChild(loader);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// Fallback response templates
function getTemplateResponse(query) {
    const q = query.toLowerCase();
    
    if (appState.language === "hi") {
        if (q.includes("loan") || q.includes("karz") || q.includes("emi")) {
            return "Aapka loans ka details read ho chuka hai. Recommended plan: credit card loan aur high interest rates (30%+) wale loans ko pehle clear karein. Isse aapka total interest cost bachega. Settle karne ke liye settlement window me contact karein.";
        }
        if (q.includes("bicha") || q.includes("becha") || q.includes("kharch")) {
            return "Main aapka transaction record kar liya hai. Aapke profile memory me add ho chuka hai. Apna total income track karne ke liye Profile tab dekhein.";
        }
        if (q.includes("gst")) {
            return "GST guidelines ke anusar, 40 Lakh (goods) aur 20 Lakh (services) se kam annual turnover par GST number mandatory nahi hai. Aap Composition Scheme elect kar sakte hain jispe sirf 1% tax lagega.";
        }
        return "Namaste! Mujhe aapki baat samajh me aayi. Main aapka companion hoon. Debt structure ya business sales track karne ke liye details share karein.";
    }
    
    // English default fallback
    if (q.includes("loan") || q.includes("debt") || q.includes("emi")) {
        return "I suggest prioritizing high-interest debts first (such as credit cards or moneylenders charging >30% per year). This is the Avalanche method. Settle outstanding cards at 60-70% if experiencing genuine cashflow issues.";
    }
    if (q.includes("gst") || q.includes("tax")) {
        return "Under the new tax regime, gross income up to ₹7 Lakh pays ZERO tax under section 87A rebate. For small shops with turnover under ₹50L, ITR-4 (presumptive scheme) is recommended.";
    }
    
    return "Thank you for sharing. I've recorded these details in your local profile database. Let me know if you want to run an EMI simulation or check your debt escape roadmap.";
}


/* =============================================================
   AUTHENTICATION & THEME MODULES
   ============================================================= */

function setupAuthAndThemeListeners() {
    // Auth Overlay Tab Switching
    const tabLogin = document.getElementById("auth-tab-login");
    const tabRegister = document.getElementById("auth-tab-register");
    const authTitle = document.getElementById("auth-title");
    const authSubtitle = document.getElementById("auth-subtitle");
    const submitBtn = document.getElementById("auth-submit-btn");
    const errorMsg = document.getElementById("auth-error");
    const successMsg = document.getElementById("auth-success");
    const authForm = document.getElementById("auth-form");

    if (tabLogin && tabRegister) {
        tabLogin.addEventListener("click", () => {
            tabLogin.classList.add("active");
            tabRegister.classList.remove("active");
            authMode = "login";
            authTitle.textContent = "ArthaSathi AI";
            authSubtitle.textContent = "Empowering India's micro-entrepreneurs. Sign in to access your personal dashboard.";
            submitBtn.textContent = "Sign In";
            errorMsg.classList.add("hidden");
            successMsg.classList.add("hidden");
        });

        tabRegister.addEventListener("click", () => {
            tabRegister.classList.add("active");
            tabLogin.classList.remove("active");
            authMode = "register";
            authTitle.textContent = "Create Account";
            authSubtitle.textContent = "Register a new username and password to secure your financial information.";
            submitBtn.textContent = "Register";
            errorMsg.classList.add("hidden");
            successMsg.classList.add("hidden");
        });
    }

    // Password Eye Toggle
    const pwToggle = document.getElementById("password-toggle");
    const pwInput = document.getElementById("auth-password");
    if (pwToggle && pwInput) {
        pwToggle.addEventListener("click", () => {
            if (pwInput.type === "password") {
                pwInput.type = "text";
                pwToggle.className = "fa-solid fa-eye-slash password-toggle";
            } else {
                pwInput.type = "password";
                pwToggle.className = "fa-solid fa-eye password-toggle";
            }
        });
    }

    // Auth Form Submission
    if (authForm) {
        authForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const usernameInput = document.getElementById("auth-username");
            const passwordInput = document.getElementById("auth-password");
            const usernameVal = usernameInput.value.trim();
            const passwordVal = passwordInput.value;

            errorMsg.classList.add("hidden");
            successMsg.classList.add("hidden");
            submitBtn.disabled = true;
            submitBtn.textContent = authMode === "login" ? "Signing In..." : "Registering...";

            try {
                const endpoint = authMode === "login" ? "/login" : "/register";
                const res = await fetch(`${API_BASE}${endpoint}`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ username: usernameVal, password: passwordVal })
                });

                const data = await res.json();
                
                if (res.ok) {
                    if (authMode === "register") {
                        successMsg.textContent = "Registration successful! Signing in...";
                        successMsg.classList.remove("hidden");
                        
                        // Automatically log in
                        setTimeout(async () => {
                            try {
                                const loginRes = await fetch(`${API_BASE}/login`, {
                                    method: "POST",
                                    headers: { "Content-Type": "application/json" },
                                    body: JSON.stringify({ username: usernameVal, password: passwordVal })
                                });
                                const loginData = await loginRes.json();
                                if (loginRes.ok) {
                                    USER_ID = loginData.user_id;
                                    USERNAME = loginData.username;
                                    localStorage.setItem("arthasathi_user_id", USER_ID);
                                    localStorage.setItem("arthasathi_username", USERNAME);
                                    
                                    authForm.reset();
                                    checkAuth();
                                    loadUserProfile();
                                }
                            } catch (err) {
                                console.error(err);
                                submitBtn.disabled = false;
                                submitBtn.textContent = "Register";
                            }
                        }, 1000);
                    } else {
                        // Login successful
                        USER_ID = data.user_id;
                        USERNAME = data.username;
                        localStorage.setItem("arthasathi_user_id", USER_ID);
                        localStorage.setItem("arthasathi_username", USERNAME);
                        
                        successMsg.textContent = "Signed in successfully!";
                        successMsg.classList.remove("hidden");
                        
                        setTimeout(() => {
                            authForm.reset();
                            checkAuth();
                            loadUserProfile();
                        }, 500);
                    }
                } else {
                    submitBtn.disabled = false;
                    submitBtn.textContent = authMode === "login" ? "Sign In" : "Register";
                    errorMsg.textContent = data.detail || "Authentication failed. Please try again.";
                    errorMsg.classList.remove("hidden");
                }
            } catch (err) {
                console.error(err);
                submitBtn.disabled = false;
                submitBtn.textContent = authMode === "login" ? "Sign In" : "Register";
                errorMsg.textContent = "Server connection error. Please try again.";
                errorMsg.classList.remove("hidden");
            }
        });
    }

    // Theme Switcher Button
    const themeBtn = document.getElementById("theme-toggle-btn");
    if (themeBtn) {
        themeBtn.addEventListener("click", () => {
            const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
            const newTheme = currentTheme === "light" ? "dark" : "light";
            setTheme(newTheme);
        });
    }

    // Logout Button
    const logoutBtn = document.getElementById("logout-btn");
    if (logoutBtn) {
        logoutBtn.addEventListener("click", () => {
            USER_ID = "";
            USERNAME = "";
            localStorage.removeItem("arthasathi_user_id");
            localStorage.removeItem("arthasathi_username");
            
            // Reset state
            appState.income = 25000;
            appState.debts = [];
            appState.transactions = [];
            
            // Clear tables & UI
            updateProfileUI();
            renderTransactions();
            
            // Reset chat messages to welcome message
            const container = document.getElementById("chat-messages");
            if (container) {
                container.innerHTML = `
                    <div class="message assistant">
                        <div class="message-avatar"><i class="fa-solid fa-robot"></i></div>
                        <div class="message-content">
                            <p>Namaste! Main ArthaSathi hoon — aapka financial dost. Main aapke loans, debt settlement, business pricing aur tax computation me madad kar sakta hoon. Aap mujhse Hindi, English ya kisi aur regional language me baat kar सकते हैं।</p>
                            <span class="message-time">Just now</span>
                        </div>
                    </div>
                `;
            }
            
            // Show auth overlay
            checkAuth();
        });
    }
}

function checkAuth() {
    const authOverlay = document.getElementById("auth-overlay");
    const headerUserInfo = document.getElementById("header-user-info");
    const headerUsername = document.getElementById("header-username");
    const profileUserId = document.getElementById("profile-user-id");
    
    if (!USER_ID) {
        if (authOverlay) authOverlay.classList.remove("hidden");
        if (headerUserInfo) headerUserInfo.classList.add("hidden");
        return false;
    } else {
        if (authOverlay) authOverlay.classList.add("hidden");
        if (headerUserInfo) headerUserInfo.classList.remove("hidden");
        if (headerUsername) headerUsername.textContent = USERNAME;
        if (profileUserId) profileUserId.textContent = `User ID: ${USERNAME}`;
        return true;
    }
}

function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
    const btn = document.getElementById("theme-toggle-btn");
    if (btn) {
        if (theme === "light") {
            btn.innerHTML = '<i class="fa-solid fa-sun"></i>';
        } else {
            btn.innerHTML = '<i class="fa-solid fa-moon"></i>';
        }
    }
}

function initTheme() {
    const savedTheme = localStorage.getItem("theme") || "dark";
    setTheme(savedTheme);
}
