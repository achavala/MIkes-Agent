import sys
import os
from pathlib import Path

# Add project root to path to import existing modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from nicegui import ui, app
from theme import Theme
import datetime

# --- Mock Data to Simulate Dashboard Content ---
tasks = [
    {"name": "Work Smarter with ClickUp AI", "priority": "High", "due": "Mon", "color": "red"},
    {"name": "Integrate Your Favorite Tools", "priority": "High", "due": "Sun", "color": "orange"},
    {"name": "Bring Your Team Onboard", "priority": "High", "due": "Sat", "color": "yellow"},
]

def layout():
    Theme.apply_global_styles()
    
    # --- HEADER ---
    with ui.header().classes(f'bg-[{Theme.BG_HEADER}] border-b border-[{Theme.BORDER_COLOR}] h-16 flex items-center px-6 gap-4'):
        # Breadcrumbs / Title
        with ui.row().classes('items-center gap-2'):
            ui.icon('cloud', color=Theme.ACCENT_PRIMARY).classes('text-2xl')
            ui.label('CloudResources InfoTech').classes('text-lg font-bold text-white')
            ui.icon('keyboard_arrow_down', color='gray').classes('cursor-pointer')

        ui.space()

        # Search Bar (Glossy)
        with ui.element('div').classes('relative'):
            ui.icon('search', color='gray').classes('absolute left-3 top-2.5')
            ui.input(placeholder='Search').classes('bg-[#1e1e1e] border border-[#3e3e3e] rounded-full pl-8 pr-4 py-1 text-sm w-64 focus:border-purple-500 transition-colors')

        # User Profile & Actions
        with ui.row().classes('items-center gap-3'):
            ui.button(icon='add').props('flat round color=white size=sm').classes('bg-gradient-to-r from-purple-500 to-pink-500')
            with ui.avatar(color='purple', text_color='white', size='md').classes('cursor-pointer'):
                ui.label('AC')
            ui.button('Manage cards').props('outline color=white size=sm').classes('rounded-lg border-[#3e3e3e]')

    # --- SIDEBAR ---
    with ui.left_drawer(value=True).classes(f'bg-[{Theme.BG_SIDEBAR}] border-r border-[{Theme.BORDER_COLOR}] px-2 py-4 flex flex-col gap-1 w-64'):
        
        def sidebar_item(icon, text, active=False):
            bg = 'bg-white/10' if active else ''
            text_color = 'text-white' if active else 'text-[#a0a0a0]'
            with ui.row().classes(f'sidebar-item w-full items-center gap-3 px-3 py-2 cursor-pointer transition-colors {bg}'):
                ui.icon(icon, color=Theme.ACCENT_PRIMARY if active else 'gray').classes('text-xl')
                ui.label(text).classes(f'text-sm font-medium {text_color}')

        ui.label('Home').classes('text-xs font-bold text-gray-500 px-3 mb-2 uppercase')
        sidebar_item('home', 'Home', active=True)
        sidebar_item('inbox', 'Inbox')
        sidebar_item('mail', 'Replies')
        
        ui.separator().classes('my-2 border-[#3e3e3e]')
        
        ui.label('My Tasks').classes('text-xs font-bold text-gray-500 px-3 mb-2 uppercase')
        sidebar_item('check_circle', 'My Tasks')
        sidebar_item('calendar_today', 'Calendar')
        
        ui.spacer()
        
        ui.label('Spaces').classes('text-xs font-bold text-gray-500 px-3 mb-2 uppercase')
        sidebar_item('folder', 'MikeAgent')
        sidebar_item('folder', 'TradeNova')

    # --- MAIN CONTENT ---
    with ui.column().classes(f'w-full h-full bg-[{Theme.BG_DARK}] p-8 gap-6'):
        
        # Greeting
        with ui.row().classes('items-center justify-between w-full'):
            ui.label(f'Good evening, Akkayya').classes('text-3xl font-bold text-white')
            ui.label(datetime.datetime.now().strftime("%A, %B %d")).classes('text-[#a0a0a0]')

        # Grid Layout
        with ui.grid(columns=2).classes('w-full gap-6'):
            
            # --- CARD 1: RECENTS ---
            with ui.column().classes(Theme.GLOSSY_CARD + ' p-0 h-64 relative overflow-hidden'):
                with ui.row().classes('w-full p-4 border-b border-[#3e3e3e] justify-between items-center'):
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('history', color='gray')
                        ui.label('Recents').classes('font-semibold text-white')
                    ui.icon('more_horiz', color='gray').classes('cursor-pointer')
                
                # Placeholder content
                with ui.column().classes('w-full p-4 gap-2'):
                    for i in range(3):
                        with ui.row().classes('w-full items-center gap-3 p-2 hover:bg-white/5 rounded'):
                            ui.icon('description', color=Theme.ACCENT_SECONDARY).classes('text-sm')
                            ui.label(f'Project Plan v{i+1}.0').classes('text-sm text-gray-300')

            # --- CARD 2: AGENDA ---
            with ui.column().classes(Theme.GLOSSY_CARD + ' p-0 h-64'):
                with ui.row().classes('w-full p-4 border-b border-[#3e3e3e] justify-between items-center'):
                    ui.label('Agenda').classes('font-semibold text-white')
                
                with ui.column().classes('w-full h-full items-center justify-center gap-4'):
                    ui.icon('calendar_month', size='3em', color='gray').classes('opacity-20')
                    ui.label('Connect your calendar to view upcoming events').classes('text-sm text-gray-500 text-center px-8')
                    with ui.row().classes('gap-2'):
                        ui.button('Google Calendar', icon='event').props('outline size=sm color=white').classes('rounded-lg')
                        ui.button('Outlook', icon='event').props('outline size=sm color=white').classes('rounded-lg')

            # --- CARD 3: MY WORK (Stats) ---
            with ui.column().classes(Theme.GLOSSY_CARD + ' p-0'):
                with ui.row().classes('w-full p-4 border-b border-[#3e3e3e] gap-4'):
                    ui.label('My Work').classes('font-semibold text-white cursor-pointer border-b-2 border-purple-500 pb-1')
                    ui.label('To Do').classes('font-semibold text-gray-500 cursor-pointer')
                    ui.label('Done').classes('font-semibold text-gray-500 cursor-pointer')
                
                with ui.row().classes('w-full p-6 justify-around'):
                    def stat_item(label, value, color):
                        with ui.column().classes('items-center'):
                            ui.label(str(value)).classes(f'text-3xl font-bold text-{color}-500')
                            ui.label(label).classes('text-xs text-gray-400 uppercase tracking-wider')
                    
                    stat_item('Overdue', 0, 'red')
                    stat_item('Next', 5, 'purple')
                    stat_item('Unscheduled', 0, 'gray')

            # --- CARD 4: ASSIGNED TO ME ---
            with ui.column().classes(Theme.GLOSSY_CARD + ' p-0'):
                with ui.row().classes('w-full p-4 border-b border-[#3e3e3e] justify-between items-center'):
                    ui.label('Assigned to me').classes('font-semibold text-white')
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('filter_list', color='gray', size='xs')
                        ui.icon('check_circle', color='gray', size='xs')

                # List of tasks
                with ui.column().classes('w-full p-0'):
                    for task in tasks:
                        with ui.row().classes('w-full p-3 border-b border-[#3e3e3e] items-center justify-between hover:bg-white/5 cursor-pointer transition-colors'):
                            with ui.row().classes('items-center gap-3'):
                                ui.icon('radio_button_unchecked', color='gray').classes('hover:text-green-500')
                                ui.label(task['name']).classes('text-sm text-gray-300')
                            
                            with ui.row().classes('items-center gap-4'):
                                with ui.row().classes('items-center gap-1'):
                                    ui.icon('flag', color=task['color'])
                                    ui.label(task['priority']).classes(f'text-xs text-{task["color"]}-400')
                                ui.label(task['due']).classes('text-xs text-gray-500')

@ui.page('/')
def main_page():
    layout()

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title='Mike Agent Enterprise', dark=True, port=8081, reload=False)





