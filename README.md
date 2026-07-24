# AR Shoe Try-On

Try-on virtual de calzado 100% en el navegador (móvil), sin backend ni app. Detección del pie con **MediaPipe Pose Landmarker** y render del zapato 3D con **Three.js**.

## Demo en vivo

**https://3dcommercecl.github.io/ar-shoes/**

<img src="docs-qr.png" alt="QR a la demo" width="200" />

Escaneá el QR con el celular (requiere HTTPS + permiso de cámara). Apuntá la cámara a tus pies **incluyendo parte de la pierna hasta la rodilla** — el zapato virtual aparece sobre el pie en segundos, sin calibrar nada.

Parámetros de URL útiles:
- `?debug=1` — muestra los landmarks (talón/punta/tobillo) sobre el video.
- `?cpu=1` — fuerza el delegate CPU de MediaPipe (útil en iOS si hay conflicto de WebGL con Three.js).

## Estado

Demo de **validación** (Fase 1 del [ROADMAP.md](ROADMAP.md)). El tracking es MediaPipe (3 puntos reales por pie); la rotación del zapato es 2D (yaw). La pose 6DoF real (keypoints + PnP), la oclusión de pierna y el modelo propio llegan en fases posteriores, sujetas al gate de validación de demanda (G2).

## Checklist de QA en dispositivos (T1.3)

Probar en al menos 3 móviles reales y marcar:

- [ ] iPhone (Safari) — carga, detecta el pie, el zapato lo sigue
- [ ] Android (Chrome) — ídem
- [ ] Encuadre de **solo pies** (sin rodilla): ¿mantiene la detección? Si no, ¿el hint de encuadre recupera al usuario en <5s?
- [ ] Mover la cámara mientras se usa: el zapato NO salta a posiciones absurdas
- [ ] Luz baja / interior
- [ ] Pantalón largo vs. corto (frontera pierna/zapato)
- [ ] Pie izquierdo y derecho + botón de cambio manual
- [ ] iOS: si el tracking va lento o falla, probar `?cpu=1`

Reportar los fallos con foto/video para decidir si MediaPipe es suficiente o se pasa al plan B (WebAR.rocks) antes de invertir en el modelo propio.

## Correr local

No necesita build (ES modules + CDN):

```bash
npx serve .
# o
python -m http.server 8000
```

Abrir por **HTTPS o localhost** (getUserMedia lo exige). Para probar desde el celular en la misma red, usar un túnel HTTPS (p. ej. `npx localtunnel --port 8000`) o el deploy de Pages.

## Arquitectura (resumen)

| Archivo | Rol |
|---|---|
| `app.js` | Orquestador: loop de render (60fps) + loop de detección (~15Hz) |
| `pose.js` | MediaPipe Pose Landmarker → landmarks de pie (talón/punta/tobillo) |
| `filter.js` | Suavizado temporal OneEuro |
| `renderer.js` | Three.js: render del GLB sobre el feed de cámara |
| `solver.js` | PnP (svd3x3 + modelo canónico de pie) — se integra en Fase 5 |
| `models/shoe_web.glb` | Zapato optimizado para web (1.48 MB) |
| `training/` | Pipeline del modelo propio (ver ROADMAP.md, Fases 3-4) |

Ver [ROADMAP.md](ROADMAP.md) para el plan completo por fases.
