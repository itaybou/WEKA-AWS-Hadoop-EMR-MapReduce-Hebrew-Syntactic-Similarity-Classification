package utils;

import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

public class StopWordsIdentifier {
  private static final Set<String> stopWords =
      new HashSet<String>() {
        {
          add("i");
          add("me");
          add("my");
          add("myself");
          add("we");
          add("our");
          add("ours");
          add("ourselves");
          add("you");
          add("your");
          add("yours");
          add("yourself");
          add("yourselves");
          add("he");
          add("him");
          add("his");
          add("himself");
          add("she");
          add("her");
          add("hers");
          add("herself");
          add("it");
          add("its");
          add("itself");
          add("they");
          add("them");
          add("their");
          add("theirs");
          add("themselves");
          add("what");
          add("which");
          add("who");
          add("whom");
          add("this");
          add("that");
          add("these");
          add("those");
          add("am");
          add("is");
          add("are");
          add("was");
          add("were");
          add("be");
          add("been");
          add("being");
          add("have");
          add("has");
          add("had");
          add("do");
          add("does");
          add("did");
          add("doing");
          add("a");
          add("an");
          add("the");
          add("and");
          add("but");
          add("if");
          add("or");
          add("because");
          add("as");
          add("until");
          add("while");
          add("of");
          add("at");
          add("by");
          add("for");
          add("with");
          add("about");
          add("against");
          add("between");
          add("into");
          add("through");
          add("during");
          add("before");
          add("after");
          add("above");
          add("below");
          add("to");
          add("from");
          add("up");
          add("down");
          add("in");
          add("out");
          add("on");
          add("off");
          add("over");
          add("under");
          add("again");
          add("further");
          add("then");
          add("once");
          add("here");
          add("there");
          add("when");
          add("where");
          add("why");
          add("how");
          add("all");
          add("any");
          add("both");
          add("each");
          add("few");
          add("more");
          add("most");
          add("other");
          add("some");
          add("such");
          add("no");
          add("nor");
          add("not");
          add("only");
          add("own");
          add("same");
          add("so");
          add("than");
          add("too");
          add("very");
          add("s");
          add("t");
          add("can");
          add("will");
          add("just");
          add("don");
          add("should");
          add("now");
        }
      };

  public static boolean isStopWord(String word) {
    return stopWords.contains(word.toLowerCase(Locale.ROOT));
  }

  public static Set<String> getStopWords() {
    return stopWords;
  }
}
