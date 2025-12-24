"""Prediction Reason Service - Generates AI explanations for predictions"""
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from ..services.gemini_service import GeminiService

logger = logging.getLogger(__name__)


class PredictionReasonService:
    """Service for generating AI-powered explanations for gold price predictions"""

    def __init__(self, gemini_service: Optional[GeminiService] = None):
        """
        Initialize prediction reason service

        Args:
            gemini_service: Optional Gemini service instance
        """
        self.gemini_service = gemini_service or GeminiService()

    def _format_prediction_data(self, predictions: List[Dict[str, Any]]) -> str:
        """
        Format prediction data for the prompt

        Args:
            predictions: List of prediction dictionaries

        Returns:
            Formatted string
        """
        if not predictions:
            return "No historical predictions available."

        lines = []
        for pred in predictions:
            date = pred.get("date", "Unknown")
            predicted_price = pred.get("predicted_price", 0)
            actual_price = pred.get("actual_price", "N/A")
            method = pred.get("method", "Unknown")

            if actual_price == "N/A" or actual_price is None:
                lines.append(
                    f"- {date}: Predicted ${predicted_price:.2f} (Method: {method}) - Pending")
            else:
                accuracy = pred.get("accuracy_percentage", "N/A")
                lines.append(
                    f"- {date}: Predicted ${predicted_price:.2f}, "
                    f"Actual ${actual_price:.2f} (Accuracy: {accuracy}%) - Method: {method}"
                )

        return "\n".join(lines)

    def _format_news_data(self, news_info: Optional[Dict[str, Any]]) -> str:
        """
        Format news information for the prompt
        Expects aggregated sentiment metrics (not raw articles) to reduce token usage

        Args:
            news_info: News information dictionary with aggregated sentiment and headlines

        Returns:
            Formatted string
        """
        if not news_info:
            return "No recent news data available."

        lines = []

        # Sentiment information (aggregated metrics)
        combined_sentiment = news_info.get("combined_sentiment", 0)
        news_volume = news_info.get("news_volume", 0)
        sentiment_trend = news_info.get("sentiment_trend", 0)

        if news_volume > 0:
            # Determine sentiment label
            if combined_sentiment > 0.1:
                sentiment_label = "Positive"
            elif combined_sentiment < -0.1:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"

            # Determine trend direction
            if sentiment_trend > 0.05:
                trend_direction = "improving"
            elif sentiment_trend < -0.05:
                trend_direction = "declining"
            else:
                trend_direction = "stable"

            lines.append("Market News Sentiment:")
            lines.append(
                f"  - Overall Sentiment: {sentiment_label} ({combined_sentiment:.2f})")
            lines.append(f"  - News Volume: {news_volume} articles")
            lines.append(f"  - Trend: {trend_direction}")

        # Top news headlines (top 5 only to reduce tokens)
        headlines = news_info.get("headlines", [])
        if headlines:
            lines.append(f"\nTop Recent Headlines:")
            # Top 5 headlines only
            for i, headline in enumerate(headlines[:5], 1):
                # Truncate very long headlines to save tokens
                display_headline = headline[:100] + \
                    "..." if len(headline) > 100 else headline
                lines.append(f"  {i}. {display_headline}")

        return "\n".join(lines) if lines else "No recent news data available."

    def _create_prompt(
        self,
        current_price: float,
        predicted_price: float,
        prediction_date: str,
        prediction_method: str,
        historical_predictions: List[Dict[str, Any]],
        news_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create the prompt for Gemini

        Args:
            current_price: Current gold price
            predicted_price: Predicted price
            prediction_date: Date of prediction
            prediction_method: Method used for prediction
            historical_predictions: Last 10 days of predictions
            news_info: Optional news information

        Returns:
            Formatted prompt string
        """
        change = predicted_price - current_price
        change_percent = (change / current_price *
                          100) if current_price > 0 else 0

        direction = "increase" if change > 0 else "decrease"

        prompt = f"""You are a financial analyst specializing in gold price predictions. Analyze the following prediction data and provide a clear, concise explanation for why the model predicts this price movement.

CURRENT SITUATION:
- Current Gold Price: ${current_price:.2f} per troy ounce
- Predicted Price for {prediction_date}: ${predicted_price:.2f} per troy ounce
- Predicted Change: ${abs(change):.2f} ({abs(change_percent):.2f}% {direction})
- Prediction Method: {prediction_method}

HISTORICAL PREDICTIONS (Last 10 Days):
{self._format_prediction_data(historical_predictions)}

"""

        # Only include news info if it's explicitly provided and doesn't contain Alpha Vantage data
        # Note: Alpha Vantage data is excluded from being sent to Gemini
        if news_info:
            # Verify news_info doesn't contain Alpha Vantage data before including
            # For safety, we'll skip news_info if it might contain Alpha Vantage data
            # Only include if it's from trusted sources (RSS feeds, NewsAPI)
            prompt += f"NEWS AND SENTIMENT DATA:\n{self._format_news_data(news_info)}\n\n"

        prompt += """TASK:
Provide a clear, user-friendly explanation (2-4 bullet points) explaining the key factors that likely influenced this prediction. Focus on:
1. Technical/market factors (price trends, volatility patterns)
2. News sentiment and market sentiment (if available)
3. Model confidence based on historical accuracy
4. Key market indicators

Format your response as concise bullet points that a non-technical user can understand. Use clear, professional language. Keep each point to 1-2 sentences maximum.

CRITICAL REQUIREMENTS:
- Complete ALL sentences fully - never cut off mid-sentence
- Finish ALL thoughts completely - every sentence must end with proper punctuation (. ! or ?)
- Ensure each bullet point is a complete, coherent statement
- Do not end with incomplete phrases, conjunctions (or, and, but), or cut-off text
- Your response must end with a complete sentence that makes sense on its own
- If you run out of space, finish the current sentence completely before stopping"""

        return prompt

    def _create_system_instruction(self) -> str:
        """Create system instruction for Gemini"""
        return """You are a financial analyst assistant specializing in gold price predictions. 
Your role is to explain ML model predictions in clear, accessible language for traders and investors.
Always be factual, concise, and focus on actionable insights. Avoid overly technical jargon unless necessary.

IMPORTANT: Always provide complete, fully-formed sentences. Never truncate or cut off your response mid-sentence. Ensure every thought is finished completely."""

    def generate_prediction_reasons(
        self,
        current_price: float,
        predicted_price: float,
        prediction_date: str,
        prediction_method: str,
        historical_predictions: List[Dict[str, Any]],
        news_info: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Generate AI-powered reasons for a prediction

        Args:
            current_price: Current gold price
            predicted_price: Predicted price
            prediction_date: Date of prediction
            prediction_method: Method used for prediction
            historical_predictions: Last 10 days of predictions
            news_info: Optional news information

        Returns:
            Generated explanation text or None on error
        """
        if not self.gemini_service.is_available():
            logger.warning(
                "Gemini service not available. Cannot generate prediction reasons.")
            logger.debug(
                f"API key configured: {self.gemini_service.api_key is not None}")
            return None

        try:
            prompt = self._create_prompt(
                current_price=current_price,
                predicted_price=predicted_price,
                prediction_date=prediction_date,
                prediction_method=prediction_method,
                historical_predictions=historical_predictions,
                news_info=news_info
            )

            system_instruction = self._create_system_instruction()

            logger.info("Generating prediction reasons using Gemini AI...")
            reasons = self.gemini_service.generate_text(
                prompt=prompt,
                system_instruction=system_instruction
            )

            if reasons:
                # Check if response appears incomplete (ends with incomplete sentence)
                reasons_clean = reasons.strip()
                if reasons_clean:
                    # Check for common incomplete sentence patterns
                    incomplete_indicators = []

                    # Check for incomplete words at the end
                    last_word = reasons_clean.split(
                    )[-1].lower() if reasons_clean.split() else ""
                    incomplete_words = ['or', 'and', 'but', 'that', 'which', 'when', 'where',
                                        'who', 'what', 'how', 'why', 'if', 'while', 'because', 'since', 'although']

                    if last_word in incomplete_words:
                        incomplete_indicators.append(
                            f"ends with incomplete word: '{last_word}'")

                    # Check if last sentence doesn't end with proper punctuation
                    last_char = reasons_clean[-1] if reasons_clean else ""
                    if last_char not in '.!?':
                        incomplete_indicators.append(
                            f"missing ending punctuation (ends with '{last_char}')")

                    # Check if last sentence is very short (might be cut off)
                    last_sentence = reasons_clean.split(
                        '.')[-1].strip() if '.' in reasons_clean else reasons_clean
                    if len(last_sentence) < 10 and last_sentence:
                        incomplete_indicators.append(
                            "last sentence appears too short")

                    if incomplete_indicators:
                        logger.warning(
                            f"Response appears incomplete. Indicators: {', '.join(incomplete_indicators)}")
                        logger.debug(
                            f"Last 100 chars: ...{reasons_clean[-100:]}")

                        # Try to complete the response by making a follow-up request
                        max_completion_attempts = 2
                        for attempt in range(max_completion_attempts):
                            try:
                                completion_prompt = f"""The previous gold price prediction analysis was cut off mid-sentence. Please complete ONLY the last incomplete thought:

{reasons_clean}

IMPORTANT: 
- Complete the last incomplete sentence only
- Make it a complete, coherent statement
- Do not repeat the entire analysis
- End with proper punctuation"""

                                completion = self.gemini_service.generate_text(
                                    prompt=completion_prompt,
                                    system_instruction="You are completing an incomplete sentence. Provide only the completion, making it a full, complete sentence that ends with proper punctuation."
                                )

                                if completion:
                                    completion_clean = completion.strip()
                                    # Remove any duplicate text that might have been included
                                    if completion_clean.lower().startswith(reasons_clean[-50:].lower()):
                                        completion_clean = completion_clean[len(
                                            reasons_clean[-50:]):].strip()

                                    reasons = reasons_clean + " " + completion_clean
                                    logger.info(
                                        f"Successfully completed truncated response (attempt {attempt + 1})")
                                    break
                                else:
                                    logger.warning(
                                        f"Completion attempt {attempt + 1} returned empty")
                            except Exception as completion_error:
                                logger.warning(
                                    f"Completion attempt {attempt + 1} failed: {completion_error}")
                                if attempt == max_completion_attempts - 1:
                                    logger.error(
                                        "All completion attempts failed. Returning incomplete response.")

                logger.info(
                    f"Successfully generated prediction reasons (length: {len(reasons)} chars)")
                return reasons
            else:
                logger.warning("Failed to generate prediction reasons")
                return None

        except Exception as e:
            logger.error(
                f"Error generating prediction reasons: {e}", exc_info=True)
            return None
