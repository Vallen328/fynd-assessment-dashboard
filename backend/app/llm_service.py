from google import genai
from app.config import GOOGLE_GEMINI_API_KEY
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.api_key = GOOGLE_GEMINI_API_KEY
        self.client = None
        self.model = "gemini-2.5-flash"
        
        # Only initialize client if API key is provided
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}. Using fallback responses.")
                self.client = None

    def generate_user_response(self, rating: int, review: str) -> str:
        """Generate a user-facing AI response to the review"""
        if not self.client:
            # Fallback response when API key is not available
            if rating >= 4:
                return "Thank you for your positive feedback! We're delighted that you're satisfied with our service. Your kind words motivate us to continue delivering excellence."
            elif rating == 3:
                return "Thank you for your feedback. We appreciate your insights and will use them to improve our service. Please let us know if there's anything we can do to enhance your experience."
            else:
                return "Thank you for bringing this to our attention. We sincerely apologize for your experience and would love the opportunity to make things right. Please reach out to our support team directly."
        
        try:
            prompt = f"""You are a helpful customer service AI. A user has submitted the following feedback:

Rating: {rating}/5
Review: {review}

Generate a warm, empathetic, and professional response to acknowledge their feedback. Keep it concise (2-3 sentences). 
If the review is negative, offer to help resolve the issue. If positive, thank them for their kind words."""

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0.9
                }
            )
            
            if response.text:
                return response.text.strip()
            return "Thank you for your feedback!"
        except Exception as e:
            logger.error(f"Error generating user response: {e}")
            return "Thank you for your feedback! We appreciate your input."

    def generate_summary(self, review: str) -> str:
        """Generate a summary of the review for admin dashboard"""
        if not self.client:
            # Fallback: create a simple summary from review
            if len(review) > 100:
                return review[:100] + "..."
            return review
        
        try:
            prompt = f"""Summarize the following customer review in 1-2 sentences for internal use:

Review: {review}

Provide a concise summary that captures the main points."""

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0.5
                }
            )
            
            if response.text:
                return response.text.strip()
            return "Review received"
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Review received"

    def generate_recommended_actions(self, rating: int, review: str) -> str:
        """Generate recommended actions for admin to take"""
        if not self.client:
            # Fallback recommendations based on rating
            if rating >= 5:
                return "Document as positive case study and consider for testimonial"
            elif rating >= 3:
                return "Review feedback and identify improvement areas"
            else:
                return "Escalate to management and prioritize for resolution"
        
        try:
            prompt = f"""Based on the following customer review, suggest 1-2 recommended actions for the support team:

Rating: {rating}/5
Review: {review}

Provide specific, actionable recommendations (e.g., "Follow up with customer", "Escalate to management", "Document as feature request", etc.)"""

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0.6
                }
            )
            
            if response.text:
                return response.text.strip()
            return "Review and categorize feedback"
        except Exception as e:
            logger.error(f"Error generating recommended actions: {e}")
            return "Review and categorize feedback"
