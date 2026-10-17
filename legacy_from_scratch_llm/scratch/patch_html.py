# patch_html.py - safe patching of api/static/index.html
import sys
from pathlib import Path

def main():
    html_path = Path("api/static/index.html")
    if not html_path.exists():
        print("Error: api/static/index.html not found")
        sys.exit(1)
        
    content = html_path.read_text(encoding="utf-8")
    
    # 1. Patch the beginning of body
    body_target = '<body>\n    <div class="glass-bg"></div>\n    <div class="app-container">'
    
    body_replacement = """<body>
    <!-- Background Apple-style Overlapping Ovals -->
    <div class="bg-ovals-container">
        <div class="bg-oval oval-1"></div>
        <div class="bg-oval oval-2"></div>
        <div class="bg-oval oval-3"></div>
    </div>
    
    <!-- Authentic Login Overlay -->
    <div id="auth-overlay" class="auth-overlay">
        <div class="auth-card">
            <div class="auth-logo">
                <i class="fa-solid fa-scale-balanced"></i>
            </div>
            <h2 id="auth-title">ArthaSathi AI</h2>
            <p id="auth-subtitle">Empowering India's micro-entrepreneurs. Sign in to access your personal dashboard.</p>
            
            <div class="auth-tabs">
                <button id="auth-tab-login" class="auth-tab-btn active">Sign In</button>
                <button id="auth-tab-register" class="auth-tab-btn">Register</button>
            </div>
            
            <form id="auth-form" class="auth-form">
                <div class="form-row">
                    <label for="auth-username">Username</label>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-user"></i>
                        <input type="text" id="auth-username" placeholder="Enter username" required autocomplete="username">
                    </div>
                </div>
                <div class="form-row">
                    <label for="auth-password">Password</label>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-lock"></i>
                        <input type="password" id="auth-password" placeholder="Enter password" required autocomplete="current-password">
                        <i id="password-toggle" class="fa-solid fa-eye password-toggle"></i>
                    </div>
                </div>
                
                <div id="auth-error" class="auth-error-msg hidden"></div>
                <div id="auth-success" class="auth-success-msg hidden"></div>
                
                <button type="submit" id="auth-submit-btn" class="submit-btn auth-btn">Sign In</button>
            </form>
        </div>
    </div>

    <div class="app-container">"""
    
    if body_target in content:
        content = content.replace(body_target, body_replacement)
    else:
        # Fallback to a simpler match
        body_target_simple = '<body>\n    <div class="glass-bg"></div>'
        body_replacement_simple = '<body>\n    <!-- Background Apple-style Overlapping Ovals -->\n    <div class="bg-ovals-container">\n        <div class="bg-oval oval-1"></div>\n        <div class="bg-oval oval-2"></div>\n        <div class="bg-oval oval-3"></div>\n    </div>'
        content = content.replace(body_target_simple, body_replacement_simple)
        print("Warning: Simple body target matching used")
        
    # 2. Patch the header for theme toggle and logout
    header_target = """            <div class="header-status">
                <div id="api-status-badge" class="status-badge checking">
                    <span class="pulse-dot"></span>
                    <span id="api-status-text">Checking systems...</span>
                </div>"""
                
    header_replacement = """            <div class="header-status">
                <!-- Theme Toggle Button -->
                <button id="theme-toggle-btn" class="theme-toggle-btn" title="Toggle Light/Dark Theme">
                    <i class="fa-solid fa-moon"></i>
                </button>
                
                <div id="api-status-badge" class="status-badge checking">
                    <span class="pulse-dot"></span>
                    <span id="api-status-text">Checking systems...</span>
                </div>
                
                <!-- User Profile & Logout -->
                <div id="header-user-info" class="header-user-info hidden">
                    <i class="fa-solid fa-user-circle"></i>
                    <span id="header-username">User</span>
                    <button id="logout-btn" class="logout-btn" title="Log Out"><i class="fa-solid fa-right-from-bracket"></i></button>
                </div>"""
                
    if header_target in content:
        content = content.replace(header_target, header_replacement)
    else:
        print("Error: Could not find header target in index.html")
        sys.exit(1)
        
    # 3. Write back
    html_path.write_text(content, encoding="utf-8")
    print("index.html patched successfully!")

if __name__ == "__main__":
    main()
