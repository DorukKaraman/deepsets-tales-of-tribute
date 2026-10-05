"""
Writes CardDatabase.cs into the current directory: a static table mapping
CardId to the effect vectors in card_db.py. The table the agents use is the
copy embedded in Bots/src/DeepSetsCore.cs; this output is for comparing against it.
"""
import card_db

def build_csharp_file():
    print("Generating CardDatabase.cs...")
    
    with open("CardDatabase.cs", "w") as f:
        f.write("using System.Collections.Generic;\n")
        f.write("using ScriptsOfTribute;\n")
        f.write("using ScriptsOfTribute.Board;\n")
        f.write("using ScriptsOfTribute.Board.Cards;\n")
        f.write("using ScriptsOfTribute.Serializers;\n\n")
        f.write("namespace Bots\n{\n")
        f.write("    public static class CardDatabase\n    {\n")
        
        f.write("        private static readonly Dictionary<CardId, float[]> Effects = new Dictionary<CardId, float[]>()\n")
        f.write("        {\n")
        
        # One initializer entry per card.
        for card_key, effects_list in card_db.CARD_EFFECTS.items():
            # Keys may be stored as strings or as CardId enums.
            enum_name = str(card_key).replace("CardId.", "").strip()
            
            # Convert python floats/ints to C# float syntax (e.g., 1.0 -> 1.0f)
            float_strings = [f"{float(val)}f" for val in effects_list]
            array_string = ", ".join(float_strings)
            
            f.write(f"            {{ (CardId){enum_name}, new float[] {{ {array_string} }} }},\n")
            
        f.write("        };\n\n")
        
        # Lookup used by FeatureExtractor in DeepSetsCore.cs.
        f.write("        public static float[] GetCardEffects(CardId cardId)\n")
        f.write("        {\n")
        f.write("            if (Effects.TryGetValue(cardId, out float[] effects))\n")
        f.write("                return effects;\n")
        f.write("\n            // Return a safe zero-vector if a card is somehow missing\n")
        # Same length as every effects entry (76).
        f.write(f"            return new float[{len(effects_list)}];\n") 
        f.write("        }\n")
        f.write("    }\n")
        f.write("}\n")
        
    print("✅ Successfully generated CardDatabase.cs!")

if __name__ == "__main__":
    build_csharp_file()