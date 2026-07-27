# Guía: grabar los videos para entrenar el modelo

> **Por qué videos y no fotos:** con SAM2 clickeás el pie/zapato/pierna **una sola vez** en el
> primer frame y las máscaras se propagan solas a todo el clip. Un video de 30 s rinde cientos de
> frames etiquetados por ~1 minuto de trabajo (contra ~2 min por foto). Es la misma estrategia de
> datos que usan Wanna y Kivisense.

---

## 1. El encuadre: esto es lo más importante

Grabá **exactamente lo que va a ver la app**: una persona parada, teléfono en mano, mirando hacia
sus propios pies. Ese encuadre "POV" es el que hizo fracasar a MediaPipe y la razón por la que
entrenamos modelo propio — así que el dataset tiene que estar dominado por él.

| Regla | Por qué |
|---|---|
| **Pies CENTRADOS en el cuadro** | El modelo (y la app) solo ven el **cuadrado central** de la imagen. Lo que quede arriba o abajo de ese cuadrado se descarta. Si los pies quedan al fondo del cuadro, se pierden. |
| **Incluí parte de la pierna / pantalón** | La frontera pierna↔zapato es lo que el modelo debe aprender para la oclusión. |
| **Teléfono en vertical**, como se usa | Coincide con el uso real. |
| **Distancia: 60 cm – 1.5 m** | Como cuando mirás tus pies parado. No macro pegado al zapato. |

**Repartí los clips entre estos ángulos:** ~40 % mirando casi a plomo hacia abajo (cenital),
~30 % en diagonal (el típico "selfie de pies"), ~20 % de frente a la altura de la rodilla,
~10 % lateral.

---

## 2. Cuántos clips y de qué duración

- **Duración ideal: 20–40 s** por clip. Clips de 10 s también sirven (rinden menos frames).
- **Meta: 20–30 clips** en total, con **≥ 5 personas** distintas.
- **Arrancá con 3–5 clips** para validar el flujo completo antes de producir en masa.
- Regla clave: **la variación va ENTRE clips, no dentro de un clip.** Cada clip = una persona, un
  calzado, un piso, una luz. Cambiá esas cosas al pasar al siguiente.

---

## 3. Reglas de oro para que SAM2 no falle

Estas cinco condicionan que la propagación automática funcione. No son opcionales:

1. **Movimientos LENTOS.** El motion blur es el enemigo n.º 1: si el frame sale movido, SAM2
   pierde el objeto y todos los frames siguientes salen mal. Movete a la mitad de la velocidad
   que te parece natural.
2. **El primer frame es sagrado.** Es donde vas a hacer los clics. Tiene que mostrar **pie,
   zapato y pierna bien visibles, nítidos y sin nada que los tape**. Si el clip empieza con el
   pie movido o cortado, no sirve.
3. **El pie nunca sale del cuadro.** Que salga y vuelva a entrar rompe el seguimiento.
4. **Luz constante dentro del clip.** No pases de sol a sombra en el mismo clip; grabá dos clips.
5. **Bloqueá exposición y foco** si tu teléfono lo permite (mantené el dedo apretado sobre la
   pantalla hasta que aparezca "AE/AF Lock"). Evita que la cámara "respire" cambiando brillo.

---

## 4. Qué variar entre clips (la diversidad es lo que da precisión)

| Variable | Qué cubrir |
|---|---|
| **Personas** | Mínimo 5 distintas: distinto tono de piel, tamaño y forma de pie |
| **Calzado** | Zapatillas, zapatos, botas, sandalias, **medias/calcetines** y **pie descalzo** |
| **Pierna** | Mitad con **pantalón** cubriendo el tobillo (largo y corto), mitad con pierna descubierta |
| **Piso** | ≥ 6 tipos: madera, cerámico, alfombra, cemento, exterior (pasto/vereda), piso oscuro |
| **Luz** | Natural, artificial cálida, mixta, contraluz suave, sombra dura del propio pie |

---

## 5. Qué hacer dentro de cada clip (20–40 s dan para todo esto)

Movimientos lentos, encadenados:

1. Quedate quieto 2 s al principio (ese es el frame de los clics).
2. Caminá **en el lugar**, despacio.
3. **Girá el pie**: punta hacia adentro, hacia afuera.
4. **Levantá el talón** (como al dar un paso) y bajalo.
5. Un paso adelante y uno atrás.
6. Movés vos la cámara: acercate y alejate un poco, y describí un arco lento alrededor del pie.

Ese repertorio le enseña al modelo el pie en todas las poses que necesita.

---

## 6. Configuración del teléfono

- **1080p a 30 fps** (no hace falta 4K; pesa el triple y no aporta).
- **Vertical.**
- **No uses el lente ultra gran angular** (el 0.5x): deforma la perspectiva y no coincide con el
  lente que usa la app.
- Limpiá el vidrio de la cámara (suena tonto; una huella arruina el clip).
- Sin filtros, sin cámara lenta, sin estabilización "cinematográfica" agresiva.

---

## 7. Dónde dejar los archivos

```
c:\Users\Iga\Documents\ar-shoes\training\data_videos\
```

Nombres **sin espacios ni acentos**: `clip01.mp4`, `clip02.mp4`, … El nombre del archivo se usa
como identificador y garantiza que todos los frames de un mismo video vayan juntos a entrenamiento
o a validación (nunca separados: si se separaran, la métrica de precisión saldría falseada).

Los videos **no se suben al repositorio** (están en `.gitignore`): quedan solo en tu máquina.

---

## 8. Cómo se procesan

```bash
cd training
py -3 2b_sam2_video.py --video data_videos/clip01.mp4 --extract_stride 3 --max_side 640
```

Se abre el primer frame y hacés los clics:

| Tecla / acción | Qué marca |
|---|---|
| **`2`** + clic | El **pie** (piel visible: empeine, tobillo) |
| **`3`** + clic | El **zapato** |
| **`1`** + clic | La **pierna / pantalón** |
| **clic derecho** | Punto negativo: corrige si la máscara se desborda |
| **`R`** | Resetear los clics |
| **ENTER** | Propagar a todo el clip y exportar |

Después, para revisar y descartar los frames donde la propagación se perdió:

```bash
py -3 2b_sam2_video.py --review --out data
```

(Cualquier tecla = siguiente frame, **`D`** = borrar ese frame, **`Q`** = salir.)

⏱️ **Tiempo:** en CPU es lento (~10 s por frame procesado). Un clip de 10 s con
`--extract_stride 3` son ~100 frames ≈ 15–20 min. Para procesar muchos clips conviene GPU
(el mismo script en Google Colab con `--device cuda` tarda un par de minutos por clip).

---

## 9. Checklist antes de grabar

- [ ] Teléfono en vertical, 1080p/30fps, lente normal (no 0.5x)
- [ ] Exposición y foco bloqueados
- [ ] Los pies quedan **en el centro** del cuadro, con parte de la pierna visible
- [ ] El clip **empieza** con 2 s quietos y el pie nítido y despejado
- [ ] Todo el clip es una sola combinación de persona/calzado/piso/luz
- [ ] Movimientos lentos, sin que el pie salga del cuadro
- [ ] Nombre `clipNN.mp4` en `training/data_videos/`

## Errores comunes

| Error | Consecuencia |
|---|---|
| Movimientos rápidos / motion blur | SAM2 pierde el objeto: el clip entero se descarta |
| Pies al borde inferior del cuadro | El recorte cuadrado central se los come |
| Primer frame movido o con el pie tapado | No se puede clickear bien: hay que descartar el clip |
| Cambiar de calzado o piso dentro del mismo clip | Los clics del frame 1 dejan de corresponder |
| Grabar con ultra gran angular (0.5x) | La deformación no coincide con la cámara de la app |
| Sol a sombra dentro del clip | La máscara se degrada al cambiar el brillo |
