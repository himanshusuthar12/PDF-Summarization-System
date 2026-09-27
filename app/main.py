import os
import uuid
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from time import perf_counter

import fitz
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

try:
    from openai import OpenAI
    HAS_OPENAI_LIB = True
except ImportError:
    HAS_OPENAI_LIB = False

try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
    HAS_SUMY = True
except ImportError:
    HAS_SUMY = False

load_dotenv()

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/application.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("PDFIntelligence")

app = FastAPI(
    title="PDF Intelligence Platform",
    version="1.0.0",
    description="Enterprise PDF summarization API",
)

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "20")) * 1024 * 1024
MAX_PAGES = int(os.getenv("MAX_PDF_PAGES", "200"))
MAX_WORDS_PER_CHUNK = int(os.getenv("MAX_WORDS_PER_CHUNK", "1200"))


class ExecutionMode(str, Enum):
    LLM = "LLM_OPENAI"
    MANUAL = "MANUAL_EXTRACTIVE"


class ProcessingStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SummaryResult(BaseModel):
    executive_summary: str
    section_breakdown: List[Dict[str, str]]
    engine_used: ExecutionMode
    processed_pages: int
    processed_sections: int = 0
    processing_time_seconds: Optional[float] = None
    fallback_triggered: bool = False


class JobStatusResponse(BaseModel):
    job_id: str
    status: ProcessingStatus
    result: Optional[SummaryResult] = None
    error: Optional[str] = None


JOBS_DB: Dict[str, Dict[str, Any]] = {}


class PDFProcessor:
    @staticmethod
    def extract_chunks(
        pdf_bytes: bytes,
        max_words: int = MAX_WORDS_PER_CHUNK,
    ) -> Tuple[List[Dict[str, str]], int]:
        doc = None
        try:
            if not pdf_bytes:
                raise ValueError("The uploaded PDF is empty.")

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            if doc.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported.")

            total_pages = len(doc)

            if total_pages == 0:
                raise ValueError("The PDF contains no pages.")

            if total_pages > MAX_PAGES:
                raise ValueError(
                    f"PDF contains {total_pages} pages. "
                    f"Maximum supported pages: {MAX_PAGES}."
                )

            sections: List[Dict[str, str]] = []
            current_title = "Document Overview"
            current_words: List[str] = []

            for page_number, page in enumerate(doc, start=1):
                try:
                    text = page.get_text("text") or ""
                except Exception as exc:
                    logger.warning(
                        "Could not extract page %s: %s",
                        page_number,
                        exc,
                    )
                    continue

                for line in text.splitlines():
                    line_str = line.strip()

                    if not line_str:
                        continue

                    if (
                        len(line_str) < 60
                        and line_str.isupper()
                        and len(line_str.split()) <= 10
                    ):
                        if current_words:
                            sections.append(
                                {
                                    "title": current_title,
                                    "content": " ".join(current_words),
                                }
                            )
                            current_words = []

                        current_title = line_str
                    else:
                        current_words.append(line_str)

            if current_words:
                sections.append(
                    {
                        "title": current_title,
                        "content": " ".join(current_words),
                    }
                )

            chunks: List[Dict[str, str]] = []

            for section in sections:
                words = section["content"].split()

                if len(words) > max_words:
                    for start in range(0, len(words), max_words):
                        part_number = start // max_words + 1
                        part_words = words[start : start + max_words]

                        chunks.append(
                            {
                                "title": (
                                    f"{section['title']} "
                                    f"(Part {part_number})"
                                ),
                                "content": " ".join(part_words),
                            }
                        )

                elif len(words) > 20:
                    chunks.append(section)

            if not chunks:
                raise ValueError(
                    "No readable text was found in the PDF. "
                    "The document may be scanned/image-only."
                )

            return chunks, total_pages

        except ValueError:
            raise
        except Exception as exc:
            logger.exception("PDF extraction failed.")
            raise RuntimeError(
                "Unable to extract text from the PDF."
            ) from exc
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass


class SummarizationPipeline:
    @staticmethod
    def run_llm_engine(
        chunks: List[Dict[str, str]],
        api_key: str,
    ) -> Tuple[str, List[Dict[str, str]]]:

        if not HAS_OPENAI_LIB:
            raise RuntimeError(
                "OpenAI package is not installed."
            )

        client = OpenAI(api_key=api_key)
        section_breakdowns: List[Dict[str, str]] = []

        for index, chunk in enumerate(chunks, start=1):
            try:
                response = client.chat.completions.create(
                    model=os.getenv(
                        "OPENAI_MODEL",
                        "gpt-4o-mini",
                    ),
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an enterprise document "
                                "summarization assistant. "
                                "Extract important facts, decisions, "
                                "risks, numbers, and key takeaways. "
                                "Return concise bullet points."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Section: {chunk['title']}\n\n"
                                f"Content:\n{chunk['content']}"
                            ),
                        },
                    ],
                    temperature=0.2,
                    max_tokens=400,
                )

                content = (
                    response.choices[0].message.content
                    if response.choices
                    else None
                )

                if not content:
                    raise RuntimeError(
                        f"Empty LLM response for chunk {index}."
                    )

                section_breakdowns.append(
                    {
                        "section": chunk["title"],
                        "summary": content.strip(),
                    }
                )

            except Exception as exc:
                logger.exception(
                    "LLM failed for chunk %s.",
                    index,
                )
                raise RuntimeError(
                    f"LLM summarization failed for section "
                    f"{chunk['title']}."
                ) from exc

        combined = "\n\n".join(
            f"### {item['section']}\n{item['summary']}"
            for item in section_breakdowns
        )

        try:
            reduce_response = client.chat.completions.create(
                model=os.getenv(
                    "OPENAI_MODEL",
                    "gpt-4o-mini",
                ),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Create a concise executive summary from "
                            "the supplied section summaries. Include "
                            "the main purpose, key findings, important "
                            "numbers, risks, and conclusions."
                        ),
                    },
                    {
                        "role": "user",
                        "content": combined[:12000],
                    },
                ],
                temperature=0.2,
                max_tokens=700,
            )

            content = (
                reduce_response.choices[0].message.content
                if reduce_response.choices
                else None
            )

            if not content:
                raise RuntimeError(
                    "Empty executive summary response."
                )

            return content.strip(), section_breakdowns

        except Exception as exc:
            logger.exception("Executive summary generation failed.")
            raise RuntimeError(
                "Unable to generate executive summary."
            ) from exc

    @staticmethod
    def run_manual_engine(
        chunks: List[Dict[str, str]],
    ) -> Tuple[str, List[Dict[str, str]]]:

        section_breakdowns: List[Dict[str, str]] = []
        all_key_sentences: List[str] = []

        for chunk in chunks:
            content = chunk["content"]

            try:
                if HAS_SUMY:
                    parser = PlaintextParser.from_string(
                        content,
                        Tokenizer("english"),
                    )
                    summarizer = LsaSummarizer(
                        Stemmer("english")
                    )
                    summarizer.stop_words = get_stop_words(
                        "english"
                    )

                    sentences = summarizer(
                        parser.document,
                        3,
                    )

                    summary_text = "\n".join(
                        f"• {str(sentence)}"
                        for sentence in sentences
                    )

                else:
                    sentences = [
                        sentence.strip()
                        for sentence in content.split(".")
                        if len(sentence.strip()) > 30
                    ]

                    summary_text = "\n".join(
                        f"• {sentence}."
                        for sentence in sentences[:3]
                    )

                if not summary_text:
                    summary_text = (
                        "• No significant extractive summary "
                        "was identified."
                    )

            except Exception as exc:
                logger.warning(
                    "Fallback summarization failed for '%s': %s",
                    chunk["title"],
                    exc,
                )
                summary_text = (
                    "• Unable to generate a summary for this section."
                )

            section_breakdowns.append(
                {
                    "section": chunk["title"],
                    "summary": summary_text,
                }
            )

            all_key_sentences.append(
                f"From {chunk['title']}:\n{summary_text}"
            )

        executive_summary = (
            "### Executive Overview "
            "(Algorithmic Extractive Engine)\n\n"
            + "\n\n".join(all_key_sentences[:6])
        )

        return executive_summary, section_breakdowns


async def process_pdf_workflow(
    job_id: str,
    pdf_bytes: bytes,
) -> None:

    start_time = perf_counter()

    if job_id not in JOBS_DB:
        logger.error(
            "Job %s disappeared before processing.",
            job_id,
        )
        return

    JOBS_DB[job_id]["status"] = ProcessingStatus.PROCESSING

    try:
        chunks, total_pages = PDFProcessor.extract_chunks(
            pdf_bytes
        )

        openai_key = os.getenv("OPENAI_API_KEY", "").strip()

        engine_used = ExecutionMode.MANUAL
        fallback_triggered = False

        exec_summary: str
        section_breakdowns: List[Dict[str, str]]

        if openai_key:
            try:
                logger.info(
                    "Job %s: starting OpenAI engine.",
                    job_id,
                )

                (
                    exec_summary,
                    section_breakdowns,
                ) = SummarizationPipeline.run_llm_engine(
                    chunks,
                    openai_key,
                )

                engine_used = ExecutionMode.LLM

            except Exception as exc:
                logger.warning(
                    "Job %s: LLM failed. Falling back. Error: %s",
                    job_id,
                    exc,
                )
                fallback_triggered = True

        if engine_used != ExecutionMode.LLM:
            logger.info(
                "Job %s: starting extractive fallback.",
                job_id,
            )

            (
                exec_summary,
                section_breakdowns,
            ) = SummarizationPipeline.run_manual_engine(
                chunks
            )

        elapsed = round(
            perf_counter() - start_time,
            2,
        )

        JOBS_DB[job_id]["status"] = (
            ProcessingStatus.COMPLETED
        )

        JOBS_DB[job_id]["result"] = SummaryResult(
            executive_summary=exec_summary,
            section_breakdown=section_breakdowns,
            engine_used=engine_used,
            processed_pages=total_pages,
            processed_sections=len(section_breakdowns),
            processing_time_seconds=elapsed,
            fallback_triggered=fallback_triggered,
        )

        logger.info(
            "Job %s completed in %ss using %s.",
            job_id,
            elapsed,
            engine_used.value,
        )

    except Exception as exc:
        logger.exception(
            "Job %s failed.",
            job_id,
        )

        JOBS_DB[job_id]["status"] = (
            ProcessingStatus.FAILED
        )

        JOBS_DB[job_id]["error"] = str(exc)


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    try:
        with open(
            "static/index.html",
            "r",
            encoding="utf-8",
        ) as file:
            return file.read()

    except FileNotFoundError as exc:
        logger.exception("UI file not found.")
        raise HTTPException(
            status_code=500,
            detail="Frontend file is not available.",
        ) from exc

    except Exception as exc:
        logger.exception("Unable to load UI.")
        raise HTTPException(
            status_code=500,
            detail="Unable to load application UI.",
        ) from exc


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "openai_available": bool(
            os.getenv("OPENAI_API_KEY", "").strip()
        ),
        "openai_package_installed": HAS_OPENAI_LIB,
        "sumy_installed": HAS_SUMY,
    }


@app.post(
    "/api/v1/summarize",
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):

    try:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Please select a PDF file.",
            )

        filename = file.filename.lower()

        if not filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported.",
            )

        if file.content_type not in (
            "application/pdf",
            "application/octet-stream",
            None,
        ):
            logger.warning(
                "Unexpected content type: %s",
                file.content_type,
            )

        pdf_bytes = await file.read()

        if len(pdf_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File is too large. Maximum allowed size "
                    f"is {MAX_FILE_SIZE // (1024 * 1024)} MB."
                ),
            )

        if not pdf_bytes.startswith(b"%PDF"):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file does not appear to be a valid PDF.",
            )

        job_id = str(uuid.uuid4())

        JOBS_DB[job_id] = {
            "status": ProcessingStatus.QUEUED,
            "result": None,
            "error": None,
        }

        background_tasks.add_task(
            process_pdf_workflow,
            job_id,
            pdf_bytes,
        )

        logger.info(
            "Job %s queued. File=%s Size=%s bytes",
            job_id,
            file.filename,
            len(pdf_bytes),
        )

        return {
            "job_id": job_id,
            "status": ProcessingStatus.QUEUED,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("PDF submission failed.")
        raise HTTPException(
            status_code=500,
            detail="Unable to submit the PDF for processing.",
        ) from exc

    finally:
        try:
            await file.close()
        except Exception:
            pass


@app.get(
    "/api/v1/jobs/{job_id}",
    response_model=JobStatusResponse,
)
async def get_status(job_id: str):

    try:
        job = JOBS_DB.get(job_id)

        if not job:
            raise HTTPException(
                status_code=404,
                detail="Job not found.",
            )

        return JobStatusResponse(
            job_id=job_id,
            status=job["status"],
            result=job.get("result"),
            error=job.get("error"),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Could not retrieve job %s.",
            job_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve job status.",
        ) from exc


@app.delete("/api/v1/jobs/{job_id}")
async def delete_job(job_id: str):

    if job_id not in JOBS_DB:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    del JOBS_DB[job_id]

    return {
        "message": "Job deleted successfully.",
        "job_id": job_id,
    }
