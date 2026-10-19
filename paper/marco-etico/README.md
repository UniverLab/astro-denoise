# Marco Ético — Proyecto Astro-Denoise

## Descripción

Este directorio contiene el documento de evaluación ética del proyecto **Astro-Denoise**, una iniciativa de investigación científica que desarrolla herramientas de acceso abierto para procesamiento de imágenes astronómicas simuladas (LSST DC2).

El documento evalúa la conformidad del proyecto con:

- **Normativa Nacional (Colombia):**
  - Ley 1581 de 2012 (Protección de Datos Personales)
  - Políticas de Ciencia Abierta de Minciencias

- **Estándares Internacionales:**
  - Principios FAIR (Findable, Accessible, Interoperable, Reusable)
  - UNESCO Recommendation on Open Science (2021)

- **Principios Éticos de Investigación:**
  - Reproducibilidad y Transparencia Científica
  - Equidad en Acceso a Recursos
  - Gobernanza Responsable de Datos
  - Excelencia Científica

## Estructura

```
astro-denoise-marco-etico/
├── main.tex                          # Archivo principal
├── project.toml                      # Configuración texforge
├── build/
│   └── main.pdf                      # Documento compilado (PDF)
├── sections/
│   ├── resumen.tex                   # Resumen ejecutivo
│   ├── introduccion.tex               # Contexto del sector astronómico
│   ├── caso_estudio.tex              # Descripción del proyecto
│   ├── riesgos_eticos.tex            # Riesgos identificados
│   ├── marco_etico.tex               # Marco normativo aplicable
│   ├── principios_aplicados.tex      # Justificación de 4+ principios
│   └── recomendaciones.tex           # Recomendaciones operativas
└── bib/
    └── referencias_eticas.bib        # Referencias bibliográficas
```

## Compilación

Requisitos: [texforge](https://github.com/JheisonMB/texforge)

```bash
# Instalar texforge (si no está instalado)
curl -fsSL https://raw.githubusercontent.com/JheisonMB/texforge/main/install.sh | sh

# Verificar proyecto
cd astro-denoise-marco-etico
texforge check

# Compilar a PDF
texforge build

# Resultado: build/main.pdf
```

## Contenido Clave

### Resumen Ejecutivo
- Propósito y hallazgos principales
- Cumplimiento de criterios éticos
- Recomendación: APROBADO

### Introducción (Contexto No-Técnico)
- ¿Qué es el denoising astronómico?
- Observatorio Vera Rubin (LSST)
- Desafío ético: acceso desigual a tecnología científica

### Caso de Estudio
- Características éticas de Astro-Denoise
- Código abierto y datos públicos
- Inclusión geográfica
- Reproducibilidad garantizada

### Riesgos Éticos Identificados
1. **Irreproducibilidad científica**
   - Mitigación: código abierto, datos versionados, reportes estructurados

2. **Brecha de acceso institucional (inequidad geográfica)**
   - Mitigación: acceso sin restricción, infraestructura mínima, idioma accesible

3. **Falta de gobernanza de datos**
   - Mitigación: licencias explícitas, documentación clara

### Marco Ético Aplicable
- Ley 1581/2012 (Colombia)
- Políticas Minciencias
- Principios FAIR
- UNESCO Open Science Recommendation

### Principios Aplicados (Justificados)
1. **Reproducibilidad y Transparencia Científica**
2. **Equidad en Acceso a Recursos Científicos**
3. **Gobernanza Responsable de Datos**
4. **Excelencia Científica**

### Recomendaciones Operativas
- Sostenibilidad institucional
- Transparencia en uso y contribuciones
- Responsabilidad y monitoreo continuo

## Criterios de Evaluación (Cumplidos)

✅ **Criterio 1:** Se identifica la aplicación del marco ético de Colombia
- Implementación de Ley 1581 (principios aplicables)
- Alineación con políticas Minciencias

✅ **Criterio 2:** Caso de estudio detallado y apropiado
- Astro-Denoise como respuesta ética a brecha de acceso
- Contexto: instrumentalización de LSST en Latinoamérica

✅ **Criterio 3:** Justificación de 2+ principios éticos
- 4 principios documentados y justificados explícitamente
- Referencias a normas internacionales y literatura

## Audiencia

- **Nivel:** Administrativo/comités de ética
- **Profundidad técnica:** Media (explicaciones sin jerga astronómica profunda)
- **Idioma:** Español

## Referencias

El documento incluye referencias explícitas a:

- Colombia (2012). Ley 1581 de Protección de Datos Personales
- Minciencias (2022). Política de Ciencia Abierta
- UNESCO (2021). Recommendation on Open Science
- Wilkinson et al. (2016). FAIR Guiding Principles
- Más referencias disponibles en `bib/referencias_eticas.bib`

## Uso del Documento

1. **Comités de Ética:** Usar como base para aprobación del proyecto
2. **Investigadores:** Referencia de principios éticos aplicables
3. **Documentación:** Registro de compromiso con reproducibilidad y acceso abierto
4. **Comunicación:** Material para demostrar responsabilidad institucional

## Mantenimiento

- **Revisión:** Anual o cuando cambien políticas regulatorias
- **Actualización:** Cuando se descubran riesgos nuevos
- **Auditoría:** Verificar cumplimiento con recomendaciones

## Contacto

**Autor:** Jheison Martinez Bolivar  
**Institución:** Universidad La Salle  
**Correo:** jhmartinez07@unisalle.edu.co

---

**Documento clasificado:** Acceso Abierto  
**Última actualización:** 2026-06-07  
**Estado:** APROBADO
