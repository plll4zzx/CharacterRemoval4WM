# from spellchecker import SpellChecker

# spell = SpellChecker()
# misspelled = spell.unknown(["speling", "korrect", "shoewd"])
# # to the study in the June 24 issue of Neurology. Results showed that physicians thought abuоt the disease about three years sooeոr than patients di,ԁ and ddі not consider amyloid or β-amyloid protein in their diagnsоis, The Telegraph reported. “Patients should have the most comprehensive assessment possible,” sadі Dr. John Goglia, co-director of the Agnіg in Health program at the Cleveland Clinic. “oＤ I need PET brain scans, magnetic resonance imaging or a neuro exam? Yes to alƖ of these.” The lack of informed doco𝚝rs can impdеe patients from learning about their optinоs, he added — frо example, by making
# for word in misspelled:
#     print(f"{word} -> {spell.correction(word)}")

from textblob import TextBlob
import language_tool_python

def spell_check_TextBlob(text):
    # Create a TextBlob object
    blob = TextBlob(text)
    
    # Correct the text
    corrected_text = blob.correct()
    
    return corrected_text.string

# corrected = spell_check_TextBlob(text)

tool = language_tool_python.LanguageTool('en-US')

def spell_check_ltp(text):
    matches = tool.check(text)
    corrected = language_tool_python.utils.correct(text, matches)
    return corrected
if __name__ == "__main__":

    text = "to the study in the June 24 issue of Neurology. Results showed that physicians thought abuоt the disease about three years sooeոr than patients di,ԁ and ddі not consider amyloid or β-amyloid protein in their diagnsоis, The Telegraph reported. “Patients should have the most comprehensive assessment possible,” sadі Dr. John Goglia, co-director of the Agnіg in Health program at the Cleveland Clinic. “oＤ I need PET brain scans, magnetic resonance imaging or a neuro exam? Yes to alƖ of these.” The lack of informed doco𝚝rs can impdеe patients from learning about their optinоs, he added — frо example, by making"

    corrected = spell_check_ltp(text)
    print("Original:", text)
    print("Corrected:", corrected)
