from nicegui import ui

# ClickUp / Enterprise Dark Theme Palette
class Theme:
    # Backgrounds
    BG_DARK = '#1e1e1e'       # Main background
    BG_SIDEBAR = '#2b2b2b'    # Sidebar background
    BG_CARD = '#2b2b2b'       # Card background (slightly lighter than main)
    BG_HEADER = '#2b2b2b'     # Header background
    
    # Accents (Glossy Purple/Pink from screenshot)
    ACCENT_PRIMARY = '#7b68ee'   # Purple
    ACCENT_SECONDARY = '#d946ef' # Pink/Magenta gradient end
    
    # Text
    TEXT_PRIMARY = '#ffffff'
    TEXT_SECONDARY = '#a0a0a0'
    
    # Borders
    BORDER_COLOR = '#3e3e3e'

    # Tailwind classes for "Glossy" look
    GLASS_EFFECT = 'backdrop-blur-md bg-opacity-80'
    GLOSSY_CARD = 'bg-[#2b2b2b] border border-[#3e3e3e] rounded-xl shadow-lg hover:shadow-xl transition-shadow duration-300'
    
    @staticmethod
    def apply_global_styles():
        ui.add_head_html('''
            <style>
                body {
                    background-color: #1e1e1e;
                    color: white;
                    font-family: 'Inter', sans-serif;
                }
                .glossy-gradient {
                    background: linear-gradient(135deg, #7b68ee 0%, #d946ef 100%);
                }
                .sidebar-item:hover {
                    background-color: rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                    color: white !important;
                }
                .q-field__native, .q-field__prefix, .q-field__suffix, .q-field__input {
                    color: white !important;
                }
                .q-field__label {
                    color: #a0a0a0 !important;
                }
            </style>
        ''')





