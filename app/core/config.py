from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, populate_by_name=True)

    app_name: str = Field(default="doc-intel-engine", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    glog_minloglevel: int = Field(default=2, alias="GLOG_minloglevel")
    upload_dir: Path = Field(default=Path("./data/uploads"), alias="UPLOAD_DIR")
    max_upload_size_mb: int = Field(default=25, alias="MAX_UPLOAD_SIZE_MB")
    categories: str = Field(default="invoice,receipt,contract,resume,id_document,medical_record,bank_statement,report,letter,other", alias="CATEGORIES")
    enable_api_key_auth: bool = Field(default=False, alias="ENABLE_API_KEY_AUTH")
    api_keys: str = Field(default="", alias="API_KEYS")
    jwt_secret: str = Field(default="change-me", alias="JWT_SECRET")
    rate_limit: str = Field(default="30/minute", alias="RATE_LIMIT")
    min_direct_text_length: int = Field(default=80, alias="MIN_DIRECT_TEXT_LENGTH")
    ocr_language: str = Field(default="en", alias="OCR_LANGUAGE")
    ocr_target_languages: str = Field(default="en,bn", alias="OCR_TARGET_LANGUAGES")
    ocr_detect_orientation: bool = Field(default=True, alias="OCR_DETECT_ORIENTATION")
    paddle_pdx_disable_model_source_check: bool = Field(
        default=False, alias="PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"
    )
    ocr_device: str = Field(default="auto", alias="OCR_DEVICE")
    ocr_gpu_detection_model: str = Field(default="PP-OCRv5_server_det", alias="OCR_GPU_DETECTION_MODEL")
    ocr_gpu_recognition_model: str = Field(default="PP-OCRv4_server_rec_doc", alias="OCR_GPU_RECOGNITION_MODEL")
    ocr_cpu_detection_model: str = Field(default="PP-OCRv4_mobile_det", alias="OCR_CPU_DETECTION_MODEL")
    ocr_cpu_recognition_model: str = Field(default="en_PP-OCRv4_mobile_rec", alias="OCR_CPU_RECOGNITION_MODEL")
    image_ocr_max_dimension: int = Field(default=1600, alias="IMAGE_OCR_MAX_DIMENSION")
    image_ocr_jpeg_quality: int = Field(default=85, alias="IMAGE_OCR_JPEG_QUALITY")
    classifier_provider: str = Field(default="ollama", alias="CLASSIFIER_PROVIDER")
    classifier_model: str = Field(default="qwen2.5:3b", alias="CLASSIFIER_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_keep_alive: str = Field(default="5m", alias="OLLAMA_KEEP_ALIVE")
    vllm_base_url: str = Field(default="http://localhost:8001/v1", alias="VLLM_BASE_URL")
    vllm_api_key: str = Field(default="", alias="VLLM_API_KEY")
    classification_timeout_seconds: int = Field(default=30, alias="CLASSIFICATION_TIMEOUT_SECONDS")
    classification_max_parallel_chunks: int = Field(default=3, alias="CLASSIFICATION_MAX_PARALLEL_CHUNKS")
    classification_first_page_target_chars: int = Field(default=700, alias="CLASSIFICATION_FIRST_PAGE_TARGET_CHARS")
    classification_first_page_min_chars: int = Field(default=180, alias="CLASSIFICATION_FIRST_PAGE_MIN_CHARS")
    classification_first_page_max_chunks: int = Field(default=6, alias="CLASSIFICATION_FIRST_PAGE_MAX_CHUNKS")
    classification_first_page_batch_size: int = Field(default=3, alias="CLASSIFICATION_FIRST_PAGE_BATCH_SIZE")
    classification_early_exit_confidence: float = Field(default=0.82, alias="CLASSIFICATION_EARLY_EXIT_CONFIDENCE")
    classification_chunk_max_tokens: int = Field(default=24, alias="CLASSIFICATION_CHUNK_MAX_TOKENS")
    classification_final_max_tokens: int = Field(default=16, alias="CLASSIFICATION_FINAL_MAX_TOKENS")
    ollama_max_connections: int = Field(default=10, alias="OLLAMA_MAX_CONNECTIONS")
    text_snippet_limit: int = Field(default=6000, alias="TEXT_SNIPPET_LIMIT")
    classification_chunk_pages: int = Field(default=2, alias="CLASSIFICATION_CHUNK_PAGES")
    database_url: str = Field(default="sqlite:///./data/doc_intel.db", alias="DATABASE_URL")

    @property
    def category_list(self) -> list[str]:
        return [item.strip() for item in self.categories.split(",") if item.strip()]

    @property
    def api_key_list(self) -> list[str]:
        return [item.strip() for item in self.api_keys.split(",") if item.strip()]

    @property
    def ocr_target_language_list(self) -> list[str]:
        return [item.strip() for item in self.ocr_target_languages.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    Path("./data").mkdir(parents=True, exist_ok=True)
    return settings
