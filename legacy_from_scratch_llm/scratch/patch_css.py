# patch_css.py - safe patching of api/static/styles.css
import sys
from pathlib import Path

def main():
    css_path = Path("api/static/styles.css")
    if not css_path.exists():
        print("Error: api/static/styles.css not found")
        sys.exit(1)
        
    content = css_path.read_text(encoding="utf-8")
    
    # 1. Replace the root variables definition at the top
    vars_target = """:root {
    --bg-dark: #09090e;
    --bg-surface: rgba(18, 18, 29, 0.65);
    --bg-surface-opaque: #12121d;
    --border-color: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(99, 102, 241, 0.3);
    
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --primary-glow: rgba(99, 102, 241, 0.15);
    
    --success: #10b981;
    --success-glow: rgba(16, 185, 129, 0.1);
    
    --warning: #f59e0b;
    --danger: #ef4444;
    --danger-glow: rgba(239, 68, 68, 0.1);
    
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 20px;
    
    --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}"""

    vars_replacement = """:root {
    /* Dark Theme (Default) */
    --bg-dark: #07070a;
    --bg-surface: rgba(20, 20, 30, 0.55);
    --bg-surface-opaque: #101018;
    --border-color: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(94, 92, 230, 0.3);
    
    --primary: #5E5CE6; /* Apple Indigo */
    --primary-hover: #4c4ab2;
    --primary-glow: rgba(94, 92, 230, 0.18);
    
    --success: #30D158; /* Apple Green */
    --success-glow: rgba(48, 209, 88, 0.12);
    
    --warning: #FF9F0A; /* Apple Orange */
    --danger: #FF453A; /* Apple Red */
    --danger-glow: rgba(255, 69, 58, 0.12);
    
    --text-primary: #FFFFFF;
    --text-secondary: #8E8E93;
    --text-muted: #636366;
    
    --radius-sm: 10px;
    --radius-md: 24px; /* Squircle radius */
    --radius-lg: 32px;
    
    --transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1);
    --oval-opacity: 0.18;
}

[data-theme="light"] {
    /* Light Theme variables */
    --bg-dark: #F2F2F7; /* Apple Light Gray */
    --bg-surface: rgba(255, 255, 255, 0.72);
    --bg-surface-opaque: #FFFFFF;
    --border-color: rgba(0, 0, 0, 0.08);
    --border-hover: rgba(67, 56, 202, 0.3);
    
    --primary: #4338CA; /* iOS Indigo */
    --primary-hover: #312E81;
    --primary-glow: rgba(67, 56, 202, 0.15);
    
    --success: #34C759;
    --success-glow: rgba(52, 199, 89, 0.1);
    
    --warning: #FF9500;
    --danger: #FF3B30;
    --danger-glow: rgba(255, 59, 48, 0.1);
    
    --text-primary: #1C1C1E;
    --text-secondary: #8E8E93;
    --text-muted: #AEAEB2;
    
    --oval-opacity: 0.12;
}"""

    if vars_target in content:
        content = content.replace(vars_target, vars_replacement)
    else:
        # Check if spaces or comments were different
        print("Warning: Top variables target not found exactly. Attempting loose replacement.")
        # Fallback split
        idx_end = content.find("/* Base Styles */")
        if idx_end != -1:
            content = vars_replacement + "\n\n" + content[idx_end:]
            print("Loose replacement completed successfully.")
        else:
            print("Error: Could not locate top variables block.")
            sys.exit(1)
            
    # 2. Append new design rules (Ovals, Auth Card, Theme Toggle, User Header info) to end of CSS
    new_rules = """
/* =============================================================
   APPLE DESIGN SYSTEM EXTENSIONS
   ============================================================= */

/* Overlapping Ambient Background Ovals */
.bg-ovals-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -2;
    overflow: hidden;
    pointer-events: none;
}
.bg-oval {
    position: absolute;
    border-radius: 50%;
    filter: blur(130px);
    opacity: var(--oval-opacity);
    transition: var(--transition);
}
.oval-1 {
    width: 650px;
    height: 650px;
    background: var(--primary);
    top: -250px;
    left: -200px;
}
.oval-2 {
    width: 550px;
    height: 550px;
    background: #007AFF; /* Ambient Blue */
    bottom: -150px;
    right: -150px;
}
.oval-3 {
    width: 450px;
    height: 450px;
    background: var(--primary);
    top: 35%;
    left: 55%;
    opacity: calc(var(--oval-opacity) * 0.65);
}

/* Theme Switcher Button */
.theme-toggle-btn {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    cursor: pointer;
    font-size: 1.1rem;
    transition: var(--transition);
}
.theme-toggle-btn:hover {
    background: var(--primary-glow);
    border-color: var(--primary);
    transform: scale(1.05);
}
[data-theme="light"] .theme-toggle-btn {
    background: rgba(0, 0, 0, 0.03);
}

/* Header User Details & Logout */
.header-user-info {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--border-color);
    padding: 6px 12px;
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
    font-size: 0.85rem;
}
[data-theme="light"] .header-user-info {
    background: rgba(0, 0, 0, 0.02);
}
.header-user-info i {
    color: var(--primary);
    font-size: 1.05rem;
}
.header-username {
    font-weight: 600;
    color: var(--text-primary);
}
.logout-btn {
    background: transparent;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    margin-left: 6px;
    padding: 2px 4px;
    transition: var(--transition);
}
.logout-btn:hover {
    color: var(--danger);
    transform: scale(1.1);
}

/* Authentic Full-Screen Login Overlay */
.auth-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0, 0, 0, 0.45);
    backdrop-filter: blur(35px);
    -webkit-backdrop-filter: blur(35px);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
    transition: var(--transition);
}
[data-theme="light"] .auth-overlay {
    background: rgba(240, 240, 245, 0.45);
}

/* Authentic Apple-style Login Card */
.auth-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 40px;
    width: 100%;
    max-width: 420px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
    animation: authFadeIn 0.45s cubic-bezier(0.16, 1, 0.3, 1);
}
[data-theme="light"] .auth-card {
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
}

@keyframes authFadeIn {
    from { opacity: 0; transform: scale(0.95) translateY(10px); }
    to { opacity: 1; transform: scale(1) translateY(0); }
}

.auth-logo {
    width: 64px;
    height: 64px;
    border-radius: 18px;
    background: linear-gradient(135deg, var(--primary), var(--primary-hover));
    display: flex;
    justify-content: center;
    align-items: center;
    color: white;
    font-size: 1.8rem;
    box-shadow: 0 10px 20px var(--primary-glow);
    margin-bottom: 5px;
}

.auth-card h2 {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
}

.auth-card p {
    font-size: 0.82rem;
    color: var(--text-secondary);
    text-align: center;
    line-height: 1.45;
    margin-top: -10px;
    margin-bottom: 10px;
}

.auth-tabs {
    display: flex;
    width: 100%;
    background: rgba(0, 0, 0, 0.15);
    border: 1px solid var(--border-color);
    padding: 4px;
    border-radius: 12px;
}
[data-theme="light"] .auth-tabs {
    background: rgba(0, 0, 0, 0.03);
}

.auth-tab-btn {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text-secondary);
    padding: 8px 12px;
    font-family: inherit;
    font-size: 0.85rem;
    font-weight: 600;
    border-radius: 8px;
    cursor: pointer;
    transition: var(--transition);
}
.auth-tab-btn.active {
    background: var(--bg-surface-opaque);
    color: var(--text-primary);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

.auth-form {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.input-wrapper {
    position: relative;
    display: flex;
    align-items: center;
}
.input-wrapper i.fa-solid {
    position: absolute;
    left: 14px;
    color: var(--text-muted);
    font-size: 0.95rem;
}
.input-wrapper input {
    width: 100%;
    padding: 12px 14px 12px 42px;
    background: rgba(0, 0, 0, 0.1);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    color: var(--text-primary);
    font-family: inherit;
    font-size: 0.9rem;
    outline: none;
    transition: var(--transition);
}
[data-theme="light"] .input-wrapper input {
    background: #FFFFFF;
}
.input-wrapper input:focus {
    border-color: var(--primary);
    box-shadow: 0 0 10px var(--primary-glow);
    background: var(--bg-surface-opaque);
}
.password-toggle {
    position: absolute;
    right: 14px;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 0.95rem;
    transition: var(--transition);
    padding: 5px;
}
.password-toggle:hover {
    color: var(--text-secondary);
}

.auth-btn {
    width: 100%;
    padding: 12px;
    border-radius: 12px;
    font-size: 0.95rem;
    margin-top: 10px;
}

.auth-error-msg {
    color: var(--danger);
    background: var(--danger-glow);
    border: 1px solid rgba(255, 69, 58, 0.15);
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 0.78rem;
    text-align: center;
}
.auth-success-msg {
    color: var(--success);
    background: var(--success-glow);
    border: 1px solid rgba(48, 209, 88, 0.15);
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 0.78rem;
    text-align: center;
}

/* Adjust light-mode specifics for dashboard readability */
[data-theme="light"] body {
    color: var(--text-primary);
}
[data-theme="light"] .logo-text h1 {
    background: linear-gradient(to right, #1c1c1e, #636366);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
[data-theme="light"] .message-content {
    background: rgba(0, 0, 0, 0.02);
}
[data-theme="light"] .message.user .message-content {
    background: rgba(67, 56, 202, 0.08);
    border-color: rgba(67, 56, 202, 0.15);
}
[data-theme="light"] .chat-messages-container {
    background: rgba(0, 0, 0, 0.02);
}
[data-theme="light"] .chat-suggestions {
    background: rgba(0, 0, 0, 0.01);
}
[data-theme="light"] .chat-input-area {
    background: rgba(0, 0, 0, 0.015);
}
[data-theme="light"] .metric-card {
    background: rgba(0, 0, 0, 0.02);
}
[data-theme="light"] .strategy-sheet {
    background: rgba(0, 0, 0, 0.015);
}
[data-theme="light"] .calc-results-block {
    background: rgba(0, 0, 0, 0.02);
}
[data-theme="light"] .tab-btn.active {
    background: rgba(67, 56, 202, 0.05);
}
"""
    
    content += new_rules
    css_path.write_text(content, encoding="utf-8")
    print("styles.css patched successfully!")

if __name__ == "__main__":
    main()
