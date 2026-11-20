# Rebuilds ../index.html from app.html + core.js + app.js
import os
here = os.path.dirname(os.path.abspath(__file__))
r = lambda f: open(os.path.join(here, f), encoding='utf-8').read()
s = r('app.html').replace('/*__CORE__*/', r('core.js')).replace('/*__APP__*/', r('app.js'))
open(os.path.join(here, '..', 'index.html'), 'w', encoding='utf-8').write(s)
print(len(s), 'bytes -> index.html')
