"""Empaquetado de resultados y registro de ejecución, sin alterar el análisis."""

import csv
import hashlib
import json
import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path


class SalidaDuplicada:
    def __init__(self, consola, archivo):
        self.consola, self.archivo = consola, archivo

    def write(self, texto):
        self.consola.write(texto)
        self.archivo.write(texto)
        self.archivo.flush()
        return len(texto)

    def flush(self):
        self.consola.flush()
        self.archivo.flush()


@contextmanager
def registrar_ejecucion(directorio, nombre):
    """Conserva consola, errores y estado incluso cuando falla una fase."""
    root = Path(directorio)
    logs = root / "registros"
    logs.mkdir(parents=True, exist_ok=True)
    inicio = datetime.now(timezone.utc)
    archivo_log = logs / f"{nombre}_{inicio.strftime('%Y%m%dT%H%M%S%fZ')}.log"
    estado_path = logs / f"estado_{nombre}.json"
    estado = {"estado": "running", "inicio_utc": inicio.isoformat(),
              "registro": archivo_log.relative_to(root).as_posix(),
              "comando": [sys.executable, *sys.argv]}

    def guardar_estado():
        estado_path.write_text(json.dumps(estado, indent=2, ensure_ascii=False), encoding="utf-8")

    guardar_estado()
    with archivo_log.open("w", encoding="utf-8") as salida:
        with redirect_stdout(SalidaDuplicada(sys.stdout, salida)), redirect_stderr(SalidaDuplicada(sys.stderr, salida)):
            try:
                yield
            except BaseException as exc:
                estado.update(estado="failed", error=f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
                raise
            else:
                estado["estado"] = "completed"
            finally:
                estado["fin_utc"] = datetime.now(timezone.utc).isoformat()
                guardar_estado()


def guardar_reproducibilidad(directorio, entradas, lector):
    """Copia exacta del código, configuración y entrada de esta ejecución."""
    carpeta = Path(directorio) / "reproducibilidad"
    carpeta.mkdir(parents=True, exist_ok=True)
    for archivo in entradas:
        archivo = Path(archivo)
        (carpeta / archivo.name).write_bytes(lector(str(archivo)))
    pruebas = Path(__file__).parent / "pruebas"
    for archivo in pruebas.glob("test_*.py"):
        destino = carpeta / "pruebas" / archivo.name
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(archivo.read_bytes())
    (carpeta / "LEEME.txt").write_text(
        "COPIA DE LOS INSUMOS DE LA EJECUCIÓN\n"
        "Incluye código, parámetros, dependencias, pruebas disponibles y entrada sin modificar.\n"
        "Para reproducir: instale requirements.txt y ejecute el comando indicado en el manifiesto\n"
        "desde esta carpeta, usando los nombres locales del archivo de datos y parámetros.\n"
        "Las respuestas climáticas usadas se conservan en ../datasets/fuentes_climaticas/.\n"
        "Para reutilizar esa caché, copie ../datasets/ a result/datasets/ antes de ejecutar.\n"
        "Sin esa copia la ejecución consulta las fuentes externas de nuevo.\n",
        encoding="utf-8")


def escribir_guia_paper(directorio):
    root = Path(directorio)
    texto = ["# Resultados para el paper", "",
        "Empiece por [RESULTADOS_PRINCIPALES.txt](RESULTADOS_PRINCIPALES.txt). "
        "El alcance y las limitaciones están en [NOTAS_METODOLOGICAS.txt](NOTAS_METODOLOGICAS.txt).", "",
        "## Material principal", "",
        "Perfiles exploratorios, comparación de escala y evolución, tendencias y movilidad observada. "
        "El modo fijo o adaptativo y su respaldo se consultan en T16–T19; fijar cuatro grupos no implica perfiles confirmados.", ""]
    for p in sorted((root / "figuras").glob("Fig*.pdf")):
        numero = int(p.name.split("_")[0][3:])
        if numero in (1, 2, 3, 4, 9, 11):
            texto.append(f"- [{p.stem}]({p.relative_to(root).as_posix()}) · "
                         f"[PNG]({p.with_suffix('.png').relative_to(root).as_posix()}) · "
                         f"[nota]({p.with_suffix('.txt').relative_to(root).as_posix()})")
    texto += ["", "### Tablas principales", ""]
    for p in sorted((root / "tablas").glob("T*.csv")):
        numero = int(p.name.split("_")[0][1:])
        if numero <= 4 or 11 <= numero <= 22 or 26 <= numero <= 29:
            texto.append(f"- [{p.stem}]({p.relative_to(root).as_posix()})")
    texto += ["",
        "## Material complementario", "",
        "ML, SHAP, indicadores descriptivos de resiliencia y resultados condicionados de largo plazo. "
        "No constituyen evidencia causal ni validación de los perfiles temporales.", ""]
    for p in sorted((root / "figuras").glob("Fig*.pdf")):
        if int(p.name.split("_")[0][3:]) in (5, 6, 7, 8, 10):
            texto.append(f"- [{p.stem}]({p.relative_to(root).as_posix()}) · "
                         f"[PNG]({p.with_suffix('.png').relative_to(root).as_posix()}) · "
                         f"[nota]({p.with_suffix('.txt').relative_to(root).as_posix()})")
    texto += ["", "### Tablas complementarias", ""]
    for p in sorted((root / "tablas").glob("T*.csv")):
        if int(p.name.split("_")[0][1:]) in [5,6,7,8,9,10,23,24,25]:
            texto.append(f"- [{p.stem}]({p.relative_to(root).as_posix()})")
    texto += ["",
        "## Archivos y reproducibilidad", "",
        "- [Validaciones de clima, k y DTW](VALIDACIONES_ADICIONALES.txt): contrastes controlados, sin asumir que los nuevos métodos mejoran los resultados.",
        "- [Libro maestro de tablas](tablas/RESULTADOS_METRICAS_Y_VALIDACIONES.xlsx).",
        "- `tablas/`: CSV, versiones legibles TXT y libro maestro XLSX con métricas y validaciones.",
        "- `datasets/`: panel analizado, datos intermedios y respuestas climáticas originales.",
        "- `reproducibilidad/`: copia del código, entrada, parámetros, dependencias y pruebas de la ejecución.",
        "- `registros/`: consola, errores y estado de las ejecuciones; `historicos/` conserva registros anteriores.",
        "- `seleccion_independiente/`: salida de la selección adaptativa ejecutada por separado; "
        "el paper usa la selección de la ejecución completa en `tablas/`.",
        "- [manifest.json](manifest.json): procedencia, configuración y hashes de la ejecución completa.",
        "- [INVENTARIO_ARCHIVOS.csv](INVENTARIO_ARCHIVOS.csv): rutas, tamaños y SHA256 de los archivos entregados.", "",
        "El inventario no se incluye a sí mismo ni incluye el manifiesto para evitar referencias circulares. "
        "El manifiesto registra el inventario y los demás archivos al terminar la ejecución. "
        "Una selección independiente posterior puede añadir archivos sin actualizar el manifiesto de la ejecución completa.", ""]
    (root / "LEEME_PAPER.md").write_text("\n".join(texto), encoding="utf-8")


def clasificar_archivo(ruta):
    """Rol editorial del archivo; no cambia su contenido ni su numeración."""
    ruta = Path(ruta)
    if ruta.parts[0] == "figuras" and ruta.stem.startswith("Fig"):
        numero = int(ruta.stem.split("_")[0][3:])
        return "Principal" if numero in (1, 2, 3, 4, 9, 11) else "Complementario"
    if ruta.parts[0] == "tablas" and ruta.stem.startswith("T"):
        numero = int(ruta.stem.split("_")[0][1:])
        return "Principal" if numero <= 4 or 11 <= numero <= 22 or 26 <= numero <= 29 else "Complementario"
    return {"datasets": "Datos", "reproducibilidad": "Reproducibilidad",
            "registros": "Registro", "seleccion_independiente": "Selección independiente",
            "tablas": "Libro o diagnóstico"}.get(ruta.parts[0], "Guía y procedencia")


def finalizar_paquete(directorio):
    """Se llama con el registro cerrado: sus bytes finales quedan auditados."""
    root = Path(directorio)
    escribir_guia_paper(root)
    archivos = sorted(p for p in root.rglob("*") if p.is_file()
                      and p.name not in ("manifest.json", "INVENTARIO_ARCHIVOS.csv")
                      and "__pycache__" not in p.parts)
    with (root / "INVENTARIO_ARCHIVOS.csv").open("w", encoding="utf-8-sig", newline="") as salida:
        writer = csv.writer(salida)
        writer.writerow(["Ruta_Relativa", "Uso", "Bytes", "SHA256"])
        for p in archivos:
            writer.writerow([p.relative_to(root).as_posix(), clasificar_archivo(p.relative_to(root)), p.stat().st_size,
                             hashlib.sha256(p.read_bytes()).hexdigest()])
    manifiesto_path = root / "manifest.json"
    if manifiesto_path.exists():
        manifiesto = json.loads(manifiesto_path.read_text(encoding="utf-8"))
        archivos.append(root / "INVENTARIO_ARCHIVOS.csv")
        manifiesto["outputs"] = [{"path": p.as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                   "kind": "result"} for p in archivos]
        manifiesto["declared_results"] = [p["path"] for p in manifiesto["outputs"]]
        manifiesto_path.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
