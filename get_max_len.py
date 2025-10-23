import numpy as np
import matplotlib.pyplot as plt

def analyze_sequence_lengths(english_texts, gloss_texts):
    """Analyze sequence lengths to determine optimal max_len"""
    
    # Calculate lengths for both English and gloss
    eng_lengths = [len(text.split()) for text in english_texts]
    gloss_lengths = [len(str(text).split()) for text in gloss_texts]
    
    # Combined analysis
    all_lengths = eng_lengths + gloss_lengths
    
    print("=== SEQUENCE LENGTH ANALYSIS ===")
    print(f"English - Min: {min(eng_lengths)}, Max: {max(eng_lengths)}, Mean: {np.mean(eng_lengths):.2f}")
    print(f"Gloss   - Min: {min(gloss_lengths)}, Max: {max(gloss_lengths)}, Mean: {np.mean(gloss_lengths):.2f}")
    print(f"Overall - Min: {min(all_lengths)}, Max: {max(all_lengths)}, Mean: {np.mean(all_lengths):.2f}")
    
    # Percentile analysis
    percentiles = [50, 75, 85, 90, 95, 99]
    for p in percentiles:
        eng_p = np.percentile(eng_lengths, p)
        gloss_p = np.percentile(gloss_lengths, p)
        all_p = np.percentile(all_lengths, p)
        print(f"{p}th percentile - English: {eng_p:.1f}, Gloss: {gloss_p:.1f}, Overall: {all_p:.1f}")
    
    # Visualization (optional)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(eng_lengths, bins=50, alpha=0.7, color='blue')
    plt.title('English Text Lengths')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 2)
    plt.hist(gloss_lengths, bins=50, alpha=0.7, color='green')
    plt.title('Gloss Text Lengths')
    plt.xlabel('Length')
    
    plt.subplot(1, 3, 3)
    plt.hist(all_lengths, bins=50, alpha=0.7, color='red')
    plt.title('All Text Lengths')
    plt.xlabel('Length')
    
    plt.tight_layout()
    plt.show()
    
    return eng_lengths, gloss_lengths, all_lengths

def get_optimal_max_length(english_texts, gloss_texts, coverage=0.95, buffer=2):
    """Calculate optimal max_len based on percentile coverage"""
    
    eng_lengths = [len(text.split()) for text in english_texts]
    gloss_lengths = [len(str(text).split()) for text in gloss_texts]
    
    # Get percentiles for both
    eng_percentile = np.percentile(eng_lengths, coverage * 100)
    gloss_percentile = np.percentile(gloss_lengths, coverage * 100)
    
    # Use the larger of the two, plus buffer for <START> <END> tokens
    max_len = int(max(eng_percentile, gloss_percentile)) + buffer
    
    # Count how many sequences would be truncated
    eng_truncated = sum(1 for length in eng_lengths if length > max_len - buffer)
    gloss_truncated = sum(1 for length in gloss_lengths if length > max_len - buffer)
    
    print(f"\n=== OPTIMAL MAX LENGTH CALCULATION ===")
    print(f"Coverage target: {coverage*100}%")
    print(f"English {coverage*100}th percentile: {eng_percentile:.1f}")
    print(f"Gloss {coverage*100}th percentile: {gloss_percentile:.1f}")
    print(f"Recommended max_len: {max_len}")
    print(f"Sequences that would be truncated:")
    print(f"  - English: {eng_truncated}/{len(eng_lengths)} ({eng_truncated/len(eng_lengths)*100:.1f}%)")
    print(f"  - Gloss: {gloss_truncated}/{len(gloss_lengths)} ({gloss_truncated/len(gloss_lengths)*100:.1f}%)")
    
    return max_len