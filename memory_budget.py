# -*- coding: utf-8 -*-
"""Cuánta memoria pide la segmentación y cuánta hay disponible.

La etapa semántica reserva tres cosas grandes a la vez sobre el ráster entero: la
imagen normalizada en float32, el acumulador de puntuaciones por clase, y la máscara
de salida. Sobre un ortomosaico de 6500 x 6500 eso ronda 1,2 GB, y en un equipo
justo la ejecución muere o el sistema empieza a paginar.

Aquí se calcula lo que va a costar antes de reservar nada, y de cuántas filas puede
ser cada bloque para que quepa en el presupuesto que se le dé.
"""

import os
import platform

BYTES_PER_MB = 1024.0 * 1024.0


def available_mb():
    """Memoria libre del sistema, en MB. Devuelve 0 si no se puede averiguar."""
    system = platform.system().lower()
    if system.startswith("win"):
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys / BYTES_PER_MB
        return 0.0

    # POSIX: MemAvailable es lo que el kernel cree que se puede pedir sin paginar.
    try:
        with open("/proc/meminfo", "r") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    try:
        return (os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / BYTES_PER_MB
    except (ValueError, OSError, AttributeError):
        return 0.0


def total_mb():
    """Memoria física total, en MB. Devuelve 0 si no se puede averiguar."""
    system = platform.system().lower()
    if system.startswith("win"):
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / BYTES_PER_MB
        return 0.0
    try:
        return (os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / BYTES_PER_MB
    except (ValueError, OSError, AttributeError):
        return 0.0


def estimate_semantic(width, height, rows, window_radius, windows_per_batch,
                      n_classes=4, bands=3, blend=True):
    """Desglose en MB de lo que reserva la segmentación sobre 'rows' filas.

    Con rows = height es el coste de procesar el ráster de una vez.
    """
    plane = float(width) * float(rows)
    win = float(window_radius) * 2.0
    detalle = {
        "imagen": plane * bands * 4 / BYTES_PER_MB,
        "puntuaciones": (plane * n_classes * 4 / BYTES_PER_MB) if blend else 0.0,
        "cobertura": (plane / BYTES_PER_MB) if blend else 0.0,
        "mascara": float(width) * float(height) / BYTES_PER_MB,
        "lote": max(1, windows_per_batch) * win * win * bands * 4 / BYTES_PER_MB,
    }
    # El lote se copia una vez al pasar por ONNX Runtime, y la prediccion vuelve con
    # una clase por pixel en vez de una banda.
    detalle["prediccion"] = detalle["lote"] / bands * n_classes
    detalle["total"] = sum(detalle.values())
    return detalle


def plan_rows(width, height, window_radius, windows_per_batch, budget_mb,
              n_classes=4, bands=3, blend=True):
    """Filas por bloque para no pasar del presupuesto. 0 = cabe entero.

    Cada bloque se lee con un margen de window_radius arriba y abajo para que las
    ventanas de su borde tengan el mismo contexto que tendrian sin trocear, asi que
    el bloque util nunca baja de 2 x window_radius: por debajo de eso el margen
    costaria mas que lo que se procesa.
    """
    entero = estimate_semantic(width, height, height, window_radius,
                               windows_per_batch, n_classes, bands, blend)
    if budget_mb <= 0 or entero["total"] <= budget_mb:
        return 0

    minimo = window_radius * 2
    fijo = (estimate_semantic(width, height, 1, window_radius, windows_per_batch,
                              n_classes, bands, blend)["mascara"]
            + estimate_semantic(width, height, 1, window_radius, windows_per_batch,
                                n_classes, bands, blend)["lote"] * (1 + n_classes / float(bands)))
    por_fila = (estimate_semantic(width, height, 2, window_radius, windows_per_batch,
                                  n_classes, bands, blend)["total"]
                - estimate_semantic(width, height, 1, window_radius, windows_per_batch,
                                    n_classes, bands, blend)["total"])
    if por_fila <= 0:
        return 0

    filas = int((budget_mb - fijo) / por_fila)
    filas -= window_radius * 2          # el margen de contexto tambien ocupa
    if filas < minimo:
        filas = minimo
    if filas >= height:
        return 0
    return filas


def describe(width, height, window_radius, windows_per_batch, n_classes=4,
             bands=3, blend=True, budget_mb=0.0):
    """Líneas para el registro: lo que cuesta, lo que hay, y qué se va a hacer."""
    entero = estimate_semantic(width, height, height, window_radius,
                               windows_per_batch, n_classes, bands, blend)
    libre = available_mb()
    total = total_mb()
    lineas = [
        "Memoria estimada para la segmentación del ráster completo:",
        "   imagen normalizada  {:>8.0f} MB".format(entero["imagen"]),
        "   puntuaciones        {:>8.0f} MB".format(entero["puntuaciones"]),
        "   cobertura           {:>8.0f} MB".format(entero["cobertura"]),
        "   máscara de salida   {:>8.0f} MB".format(entero["mascara"]),
        "   lote + predicción   {:>8.0f} MB".format(entero["lote"] + entero["prediccion"]),
        "   TOTAL               {:>8.0f} MB".format(entero["total"]),
    ]
    if total:
        lineas.append("Memoria del sistema: {:.0f} MB libres de {:.0f} MB".format(libre, total))
    else:
        lineas.append("Memoria del sistema: no se pudo consultar")
    if budget_mb > 0:
        lineas.append("Presupuesto de trabajo: {:.0f} MB".format(budget_mb))
    return lineas, entero, libre, total
