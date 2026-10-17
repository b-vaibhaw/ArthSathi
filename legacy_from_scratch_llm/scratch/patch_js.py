# patch_js.py - safe patching of api/static/app.js
import sys
from pathlib import Path

def main():
    js_path = Path("api/static/app.js")
    if not js_path.exists():
        print("Error: api/static/app.js not found")
        sys.exit(1)
        
    content = js_path.read_text(encoding="utf-8")
    
    # 1. Replace the top variable declarations and initApp
    top_target = """const API_BASE = window.location.origin;
const USER_ID = "Guest_User_1";

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
    loadUserProfile();
}"""

    top_replacement = """const API_BASE = window.location.origin;
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
}"""

    if top_target in content:
        content = content.replace(top_target, top_replacement)
    else:
        # Loose search for USER_ID
        content = content.replace('const USER_ID = "Guest_User_1";', 'let USER_ID = localStorage.getItem("arthasathi_user_id") || "";\nlet USERNAME = localStorage.getItem("arthasathi_username") || "";\nlet authMode = "login";')
        print("Warning: Loose replacement for variables top part used")
        
    # 2. Replace loadUserProfile to also load transactions
    profile_target = """// Load user profile from DB
async function loadUserProfile() {
    try {
        const res = await fetch(`${API_BASE}/user/${USER_ID}/profile`);
        if (res.ok) {
            const data = await res.json();
            appState.income = data.monthly_income || 25000;
            appState.debts = data.debts || [];
            
            // Fetch transactions
            // For now, load default if none
            updateProfileUI();
        }
    } catch (e) {
        // Fallback to defaults
        updateProfileUI();
    }
}"""

    profile_replacement = """// Load user profile from DB
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
}"""

    if profile_target in content:
        content = content.replace(profile_target, profile_replacement)
    else:
        print("Warning: Could not find loadUserProfile target directly. Attempting simple replacement.")
        # Fallback to key snippet replace
        content = content.replace('appState.debts = data.debts || [];\n            \n            // Fetch transactions\n            // For now, load default if none\n            updateProfileUI();', 'appState.debts = data.debts || [];\n            appState.transactions = data.transactions || [];\n            updateProfileUI();\n            renderTransactions();')
        
    # 3. Add listener hook inside setupEventListeners
    listener_hook_target = """// Setup listeners
function setupEventListeners() {
    // Tab Switching"""
    
    listener_hook_replacement = """// Setup listeners
function setupEventListeners() {
    // Setup Auth and Theme listeners
    setupAuthAndThemeListeners();

    // Tab Switching"""
    
    if listener_hook_target in content:
        content = content.replace(listener_hook_target, listener_hook_replacement)
    else:
        print("Error: Could not locate setupEventListeners target")
        sys.exit(1)
        
    # 4. Append new functions (setupAuthAndThemeListeners, checkAuth, setTheme, initTheme)
    new_functions = """

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
                            <p>Namaste! Main ArthaSathi hoon \u2014 aapka financial dost. Main aapke loans, debt settlement, business pricing aur tax computation me madad kar sakta hoon. Aap mujhse Hindi, English ya kisi aur regional language me baat kar सकते हैं।</p>
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
"""

    content += new_functions
    js_path.write_text(content, encoding="utf-8")
    print("app.js patched successfully!")

if __name__ == "__main__":
    main()
