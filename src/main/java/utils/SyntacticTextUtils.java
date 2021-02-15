package utils;

import org.apache.hadoop.io.Text;

import java.util.regex.Pattern;

public class SyntacticTextUtils {

    private static final Pattern PAIR_REGEX = Pattern.compile("^<.+>$");

    public static Text createLexemePair(String lexeme1, String lexeme2) {
        return new Text(String.format("<%s,%s>", lexeme1, lexeme2));
    }

    public static Text createLexemeFeatureText(String lexeme, String featureWord, String dependency) {
        return new Text(String.format("<%s,%s/%s>", lexeme, featureWord, dependency));
    }

    // <lexeme, count(L=lexeme, F=feature), count(L=lexeme)>
    public static Text createLexemeFeatureTripletText(String lexeme, long countLexemeFeature, long countLexeme) {
        return new Text(String.format("<%s,%d,%d>", lexeme, countLexemeFeature, countLexeme));
    }

    public static Text createCountText(String element, long count) {
        return new Text(String.format("<%s,%d>", element, count));
    }

    public static Text createFeatureText(String featureWord, String dependency) {
        return new Text(String.format("%s/%s", featureWord, dependency));
    }

    public static boolean isPair(Text text) {
        return PAIR_REGEX.matcher(text.toString()).matches();
    }

    public static boolean isFeature(Text text) {
        return text.toString().contains("/");
    }

    public static String[] splitPairTriplet(Text text) {
        return text.toString().replaceAll("[<>]", "").split(",");
    }
}
