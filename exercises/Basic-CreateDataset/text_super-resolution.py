from svg_turtle import SvgTurtle
import cairosvg
import sys, subprocess
import random

#_proc = subprocess.Popen(['powershell.exe', 'Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"'], stdout=sys.stdout)
#_proc.communicate()

t = SvgTurtle(512, 512)
ts = t.getscreen()
ts.bgcolor("white")

s = random.randint(4, 72)
t.write("这是一串中文", move=False, align='center', font=('华文楷体', s, 'bold'))
t.write("This is a string of English", move=False, align='center', font=('华文楷体', s, 'bold'))
t.save_as('demo.svg')

cairosvg.svg2png(url='demo.svg', write_to='demo.jpeg', output_width=2048, output_height=2048, dpi=2000)
