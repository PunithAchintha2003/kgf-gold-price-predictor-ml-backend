"""Gemini API Service"""
import logging
import time
from typing import Optional, Dict, Any, List
import requests
from ..config import ai_config

# Try to import Google Generative AI SDK
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Google Gemini API"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini service

        Args:
            api_key: Optional API key. If not provided, uses config.
        """
        self.api_key = api_key or ai_config.gemini_api_key
        self.model = ai_config.gemini_model
        self.temperature = ai_config.gemini_temperature
        self.max_tokens = ai_config.gemini_max_tokens
        self.api_base = ai_config.gemini_api_base
        self.timeout = ai_config.request_timeout
        self.max_retries = ai_config.max_retries

        if not self.api_key:
            logger.warning(
                "Gemini API key not configured. AI features will be unavailable.")

    def _get_api_url(self) -> str:
        """Get the API URL for the current model"""
        # The API endpoint format is: {api_base}/models/{model}:generateContent
        # For some APIs, we might need to use the full model path
        return f"{self.api_base}/models/{self.model}:generateContent"

    def _list_available_models(self) -> Optional[List[str]]:
        """List available models from the API"""
        if not self.api_key:
            return None

        try:
            url = f"{self.api_base}/models"
            headers = {"Content-Type": "application/json"}
            response = requests.get(
                url,
                headers=headers,
                params={"key": self.api_key},
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                models = [model.get("name", "").split("/")[-1]
                          for model in data.get("models", [])]
                return models
        except Exception as e:
            logger.debug(f"Could not list models: {e}")
        return None

    def _make_request(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Make a request to Gemini API

        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction
            retry_count: Current retry attempt

        Returns:
            API response or None on error
        """
        if not self.api_key:
            logger.error("Gemini API key not configured")
            return None

        url = self._get_api_url()
        headers = {
            "Content-Type": "application/json",
        }

        # Build request payload
        contents = [{"parts": [{"text": prompt}]}]

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            }
        }

        # systemInstruction is supported in v1beta API
        if system_instruction and "v1beta" in self.api_base:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
        elif system_instruction:
            # For v1 API, prepend system instruction to prompt
            final_prompt = f"{system_instruction}\n\n{prompt}"
            contents = [{"parts": [{"text": final_prompt}]}]
            payload["contents"] = contents

        try:
            response = requests.post(
                url,
                headers=headers,
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limit or quota exceeded
                try:
                    error_data = response.json() if response.text else {}
                    error_info = error_data.get("error", {})
                    error_msg = error_info.get("message", response.text)

                    # Check if it's a quota error (not just rate limit)
                    if "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
                        logger.error(
                            f"Gemini API quota exceeded: {error_msg}. "
                            f"Free tier allows 20 requests/day per model. "
                            f"Please wait or upgrade your plan.")
                        return None

                    # Extract retry delay from error if available
                    retry_delay = None
                    if "retry_delay" in str(error_data):
                        try:
                            retry_info = error_info.get("retry_delay", {})
                            if isinstance(retry_info, dict):
                                retry_delay = retry_info.get("seconds", None)
                        except:
                            pass

                    if retry_count < self.max_retries:
                        wait_time = retry_delay if retry_delay else (
                            (2 ** retry_count) * 2)
                        logger.warning(
                            f"Rate limited. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        return self._make_request(prompt, system_instruction, retry_count + 1)
                    else:
                        logger.error(
                            "Rate limit exceeded. Max retries reached.")
                        return None
                except Exception as parse_error:
                    # If we can't parse the error, use default behavior
                    if retry_count < self.max_retries:
                        wait_time = (2 ** retry_count) * 2
                        logger.warning(
                            f"Rate limited. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        return self._make_request(prompt, system_instruction, retry_count + 1)
                    else:
                        logger.error(
                            "Rate limit exceeded. Max retries reached.")
                        return None
            elif response.status_code == 404:
                # Model not found - try fallback models
                try:
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get("error", {}).get(
                        "message", response.text)
                    logger.warning(
                        f"Model {self.model} not found: {error_msg}")

                    # Try to list available models and use the first suitable one
                    if retry_count == 0:
                        available_models = self._list_available_models()
                        if available_models:
                            # Try models in order of preference
                            preferred_models = [
                                "gemini-1.5-flash",
                                "gemini-1.5-pro",
                                "gemini-pro",
                                "gemini-1.5-flash-latest",
                                "gemini-1.5-pro-latest"
                            ]

                            for preferred in preferred_models:
                                if preferred in available_models:
                                    logger.info(
                                        f"Trying available model: {preferred}")
                                    original_model = self.model
                                    self.model = preferred
                                    result = self._make_request(
                                        prompt, system_instruction, retry_count + 1)
                                    if result:
                                        logger.info(
                                            f"Successfully used model: {preferred}")
                                        return result
                                    self.model = original_model

                            # If no preferred model works, try the first available model
                            if available_models:
                                first_model = available_models[0]
                                logger.info(
                                    f"Trying first available model: {first_model}")
                                original_model = self.model
                                self.model = first_model
                                result = self._make_request(
                                    prompt, system_instruction, retry_count + 1)
                                if result:
                                    logger.info(
                                        f"Successfully used model: {first_model}")
                                    return result
                                self.model = original_model
                        else:
                            # Fallback: try common model names
                            logger.warning(
                                "Could not list available models, trying common fallbacks")
                            fallback_models = [
                                "gemini-1.5-flash", "gemini-pro", "gemini-1.5-pro"]
                            for fallback in fallback_models:
                                logger.info(f"Trying fallback: {fallback}")
                                original_model = self.model
                                self.model = fallback
                                result = self._make_request(
                                    prompt, system_instruction, retry_count + 1)
                                if result:
                                    return result
                                self.model = original_model

                    logger.error(
                        f"All model attempts failed. Model: {self.model}, API Base: {self.api_base}")
                except Exception as e:
                    logger.error(f"Error handling 404: {e}", exc_info=True)
                return None
            else:
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                return None

        except requests.exceptions.Timeout:
            logger.error(f"Request timeout after {self.timeout} seconds")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return None

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate text using Gemini API

        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction

        Returns:
            Generated text or None on error
        """
        # Try using Google Generative AI SDK first (more reliable)
        if GENAI_AVAILABLE and self.api_key:
            try:
                # Configure the API key
                genai.configure(api_key=self.api_key)
                logger.debug(
                    f"Configured Gemini API with key (length: {len(self.api_key)})")

                # Try to find an available model by listing models first
                model = None
                try:
                    # List all available models
                    available_models = [m.name for m in genai.list_models()
                                        if 'generateContent' in m.supported_generation_methods]
                    if available_models:
                        logger.info(
                            f"Found {len(available_models)} available models")
                        # Extract model names (remove 'models/' prefix if present)
                        model_names = [
                            m.split('/')[-1] if '/' in m else m for m in available_models]
                        logger.debug(f"Available model names: {model_names}")

                        # Try preferred models in order
                        # Note: Free tier has 20 requests/day per model, so we try different models
                        preferred_models = [
                            self.model,  # Try configured model first
                            'gemini-1.5-pro',  # Try different model if quota exceeded on flash
                            'gemini-pro',  # Older model, might have different quota
                            'gemini-1.5-flash',
                            'gemini-1.5-pro-latest',
                            'gemini-1.5-flash-latest'
                        ]

                        for preferred in preferred_models:
                            if preferred in model_names:
                                try:
                                    model = genai.GenerativeModel(preferred)
                                    logger.info(f"Using model: {preferred}")
                                    break
                                except Exception as e:
                                    logger.debug(
                                        f"Could not create model {preferred}: {e}")
                                    continue

                        # If no preferred model worked, try the first available
                        if not model and model_names:
                            try:
                                first_model = model_names[0]
                                model = genai.GenerativeModel(first_model)
                                logger.info(
                                    f"Using first available model: {first_model}")
                            except Exception as e:
                                logger.warning(
                                    f"Could not use first available model: {e}")
                except Exception as list_error:
                    logger.warning(
                        f"Could not list models: {list_error}, trying direct model creation")
                    # Fallback: try direct model creation
                    for model_name in ['gemini-1.5-pro', 'gemini-pro', 'gemini-1.5-flash']:
                        try:
                            model = genai.GenerativeModel(model_name)
                            logger.info(
                                f"Successfully created model: {model_name}")
                            break
                        except:
                            continue

                if not model:
                    raise Exception(
                        "Could not find any available Gemini model")

                # Configure generation
                # Use GenerationConfig type if available, otherwise use dict
                try:
                    from google.generativeai.types import GenerationConfig
                    generation_config = GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    )
                except ImportError:
                    # Fallback to dict format
                    generation_config = {
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                    }

                # Generate content
                if system_instruction:
                    # Combine system instruction with prompt
                    full_prompt = f"{system_instruction}\n\n{prompt}"
                    response = model.generate_content(
                        full_prompt,
                        generation_config=generation_config
                    )
                else:
                    response = model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )

                if response:
                    # Handle different response types from SDK
                    text = None

                    # Try direct text attribute first (most common)
                    if hasattr(response, 'text') and response.text:
                        text = response.text
                    # Try candidates structure
                    elif hasattr(response, 'candidates') and response.candidates:
                        if len(response.candidates) > 0:
                            candidate = response.candidates[0]
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                if len(candidate.content.parts) > 0:
                                    part = candidate.content.parts[0]
                                    if hasattr(part, 'text'):
                                        text = part.text

                    if text:
                        # Check if response was truncated
                        finish_reason = None
                        if hasattr(response, 'candidates') and response.candidates:
                            candidate = response.candidates[0]
                            if hasattr(candidate, 'finish_reason'):
                                finish_reason = candidate.finish_reason
                                if finish_reason and finish_reason != 'STOP':
                                    if finish_reason == 'MAX_TOKENS':
                                        logger.warning(
                                            f"Response truncated due to MAX_TOKENS limit ({self.max_tokens}). "
                                            f"Response length: {len(text)} chars. Consider increasing max_tokens.")
                                    else:
                                        logger.warning(
                                            f"Response may be incomplete. Finish reason: {finish_reason}. "
                                            f"Response length: {len(text)} chars")
                        else:
                            logger.debug(
                                f"Generated text (length: {len(text)} chars, finish_reason: {finish_reason or 'STOP'})")

                        logger.info(
                            f"Successfully generated text using Google Generative AI SDK (length: {len(text)} chars)")
                        return text.strip()
                    else:
                        logger.warning(
                            f"Response received but no text found. Response type: {type(response)}")
                        # Check for blocking reasons
                        if hasattr(response, 'prompt_feedback'):
                            logger.warning(
                                f"Prompt feedback: {response.prompt_feedback}")
                        raise Exception("No text in response from Gemini API")

            except Exception as e:
                error_msg = str(e)

                # Check for quota/rate limit errors - don't retry if quota exceeded
                if "429" in error_msg or "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
                    # Extract retry delay if available
                    retry_delay = None
                    if "retry in" in error_msg.lower() or "retry_delay" in error_msg.lower():
                        import re
                        delay_match = re.search(
                            r'(\d+\.?\d*)\s*seconds?', error_msg, re.IGNORECASE)
                        if delay_match:
                            # Add 5 seconds buffer
                            retry_delay = int(float(delay_match.group(1))) + 5

                    logger.error(
                        f"Gemini API quota/rate limit exceeded. "
                        f"Free tier limit: 20 requests/day per model. "
                        f"{f'Retry after {retry_delay}s. ' if retry_delay else ''}"
                        f"Consider upgrading your plan or waiting before retrying.")

                    # Don't fall back to REST API if quota is exceeded - it will also fail
                    # Return None to indicate quota issue
                    return None

                logger.warning(
                    f"Google Generative AI SDK failed: {error_msg}, falling back to REST API")
                # Log more details for debugging
                if "404" in error_msg or "not found" in error_msg.lower():
                    logger.info(
                        "Model not found error - will try REST API with model discovery")

        # Fallback to REST API
        response = self._make_request(prompt, system_instruction)

        if not response:
            return None

        try:
            # Extract text from response
            candidates = response.get("candidates", [])
            if not candidates:
                logger.warning("No candidates in response")
                return None

            candidate = candidates[0]
            finish_reason = candidate.get("finishReason", "")

            # Check if response was truncated
            if finish_reason and finish_reason != "STOP":
                if finish_reason == "MAX_TOKENS":
                    logger.warning(
                        f"Response truncated due to MAX_TOKENS limit ({self.max_tokens}). "
                        f"Consider increasing GEMINI_MAX_TOKENS in .env file.")
                else:
                    logger.warning(
                        f"Response may be incomplete. Finish reason: {finish_reason}")

            content = candidate.get("content", {})
            parts = content.get("parts", [])
            if not parts:
                logger.warning("No parts in content")
                return None

            text = parts[0].get("text", "")
            if text:
                logger.debug(
                    f"Generated text via REST API (length: {len(text)} chars, finish_reason: {finish_reason or 'STOP'})")
            return text.strip() if text else None

        except Exception as e:
            logger.error(f"Error parsing response: {e}", exc_info=True)
            return None

    def is_available(self) -> bool:
        """Check if the service is available"""
        return self.api_key is not None and len(self.api_key.strip()) > 0
