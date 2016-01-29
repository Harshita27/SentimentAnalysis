/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package sentimentanalysis;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 *
 * @author Harshita Naive Bayes text categorization system to predict whether
 * movie reviews are positive or negative. The data for this “sentiment
 * analysis” task were first assembled and published in Bo Pang and Lillian Lee,
 * “A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization
 * Based on Minimum Cuts”, Proceedings of the Association for Computational
 * Linguistics, 2004.
 *
 * The program implements a bag-of-words, naive Bayes classifier using add-1
 * (Laplace) smoothing.
 */
/**
 * ****************************************************************************
 * nbtrain.java: This file performs the following tasks: 1) Recursively reads in
 * the training data as provided, divided into "pos" and "neg" reviews 2) As
 * each review file is read, the terms in the files along with their frequency
 * is stored in a Map of terms 3) Each categories length is also stored along
 * with (i e Number of files in each category) 4) The vocabulary length is also
 * calculated 5) Finally, the conditional probability of each term belonging to
 * a particular category is calculated and written to a model file 6) The model
 * file is used by the nbtest.java file to categorize the test data.
 */
/**
 * ***************************************************************************
 * This class's object holds all the necessary data used to compute the
 * conditional probabilities.
 * ***************************************************************************
 */
class trainData {

    public double posLength;
    public double negLength;
    public double vocabLength;
    public double postermLength = 0.0;
    public double negtermLength = 0.0;
    public Map<String, Double> posTermFreq = new HashMap<String, Double>();
    public Map<String, Double> negTermFreq = new HashMap<String, Double>();
    public Map<String, Double> logProbPos = new HashMap<String, Double>();
    public Map<String, Double> logProbNeg = new HashMap<String, Double>();

}

/**
 * ***************************************************************************
 * This class's object holds all the necessary data used to compute the
 * vocabulary data.
 * ***************************************************************************
 */
class vocabData {

    public int vocabLength;
    public ArrayList<String> termSkip = new ArrayList<>();
    public ArrayList<String> termSkipPos = new ArrayList<>();
    public ArrayList<String> termSkipNeg = new ArrayList<>();
}

/**
 * ***************************************************************************
 * Main Class.
 * ***************************************************************************
 */
public class nbtrain {

    static int totalDoc = 0;

    public static void main(String args[]) throws FileNotFoundException {

        // Read in the train directory
        File folder = new File(args[0]);
                ///"C:\\Users\\Harshita\\Documents\\NEU\\IR\\Assignment-6\\textcat\\train"
        //File[] listOfFiles = folder.listFiles();
        //PrintWriter writer = new PrintWriter(folder + "posfile.csv");
        int vocabSize = 0;
        int count = 0;

        trainData train = new trainData();
        vocabData v = new vocabData();

        trainData train2;
        //train2=countFilesInDirectory(folder, train, v);

        /**
         * *******************************************************************
         * Method to count files in the directory and store each term in each
         * file along with the frequency of the term.
         * *********************************************************************
         */
        Object[] ret = countFilesInDirectory(folder, train, v);

        train2 = (trainData) ret[0];
        vocabData v2 = (vocabData) ret[1];

        // System.out.println("~~~~~~~~~~~~~~~" + train2.negTermFreq.size());
        // System.out.println("~~~~~~~~~~~~~~~" + train2.posTermFreq.size());
        // System.out.println("calculating vocab length");
        /**
         * **********************************************************************
         * Method to calculate vocabulary length and remove terms whose combined
         * frequency in neg and pos categories is less than 5.
         * *********************************************************************
         */
        v2 = calculateVocabLength(train2.posTermFreq, train2.negTermFreq, v);
        //trainData train3 = new trainData();

        train2.posTermFreq = filterTerms(train2.posTermFreq, v2.termSkip);
        train2.negTermFreq = filterTerms(train2.negTermFreq, v2.termSkip);
        train2.vocabLength = v2.vocabLength;
       

        for (String s : train2.posTermFreq.keySet()) {
            train2.postermLength += train2.posTermFreq.get(s);
        }

        for (String s : train2.negTermFreq.keySet()) {
            train2.negtermLength += train2.negTermFreq.get(s);
        }

        /**
         * **********************************************************************
         * Method to calculate conditional probabilities of each term and write
         * the model file.
         * *********************************************************************
         */
        train2 = calculateProbabilities(train2);
        writeModelFile(train2);

    }

    /**
     * *************************************************************************
     * writeModelFile
     * @param t : Object of the trainData class that binds all the information
     * @throws FileNotFoundException
     **************************************************************************/
    public static void writeModelFile(trainData t) throws FileNotFoundException {
        PrintWriter writer = new PrintWriter("C:\\Users\\Harshita\\Documents\\NEU\\IR\\Assignment-6\\textcat\\ModelFile.txt");
        writer.println("length(vocab):" + t.vocabLength);
        writer.println("length(pos):" + t.posTermFreq.size());
        writer.println("length(neg):" + t.negTermFreq.size());
        writer.println("P(pos):" + Math.log(t.posLength / (t.negLength + t.posLength)));
        writer.println("P(neg):" + Math.log(t.negLength / (t.negLength + t.posLength)));
        for (String s : t.logProbPos.keySet()) {
            writer.print(s + ",");
            writer.println(t.logProbPos.get(s));
        }
        for (String s : t.logProbNeg.keySet()) {
            writer.print(s + ",");
            writer.println(t.logProbNeg.get(s));
        }
        writer.close();
    }

    /**
     * *************************************************************************
     * filterTerms:
     *
     * @param termFreq : The list of term frequencies for a particular category
     * @param termSkip : The list of terms to be removed from the vocabulary
     * whose frequency is less than 5.
     * @return : A modified map of terms along with the frequency
     ***************************************************************************/
    public static Map<String, Double> filterTerms(Map<String, Double> termFreq, ArrayList<String> termSkip) {

        Map<String, Double> filteredList = new HashMap<>();

        for (String s : termFreq.keySet()) {
            if (!termSkip.contains(s)) {
                filteredList.put(s, termFreq.get(s));
            }
        }
        return filteredList;
    }

    /**
     * *************************************************************************
     * calculateProbabilities:
     *
     * @param t : The object of trainData that holds all the relevant
     * information
     * @return : The object with modified values.
     * ************************************************************************
     */
    public static trainData calculateProbabilities(trainData t) {
        double posProbability = 0.0;
        double negProbability = 0.0;
        //System.out.println("vocablength:" + t.vocabLength);
        // System.out.println("vocablength:" + t.posTermFreq.size());

        // Calculate conditional probabilities for the "pos" category
        for (String s : t.posTermFreq.keySet()) {
            //  System.out.println("posterm:"+ t.posTermFreq.get(s));
            double num = (t.posTermFreq.get(s) + 1);
            double den = (t.vocabLength + t.posTermFreq.size());
            posProbability = num / den;
            t.logProbPos.put("pos:" + s, Math.log(posProbability));
        }

        // Calculate conditional probabilities for the "neg" category
        for (String s : t.negTermFreq.keySet()) {
            negProbability = (t.negTermFreq.get(s) + 1) / (t.vocabLength + t.negTermFreq.size());
            t.logProbNeg.put("neg:" + s, Math.log(negProbability));
        }
        return t;
    }

    /**
     * *************************************************************************
     * countFilesInDirectory:
     *
     * @param directory : The train directory
     * @param train : The object from trainData that holds the information for
     * all the terms
     * @param v : The object of vocabData that holds vocabulary related
     * information
     * @return : An array of objects holding trainData and vocabData
     * @throws FileNotFoundException
     * ************************************************************************
     */
    public static Object[] countFilesInDirectory(File directory, trainData train, vocabData v) throws FileNotFoundException {

        int posCount = 0;
        int negCount = 0;
        for (File file : directory.listFiles()) {

            if (file.isFile()) {
                String dirName = file.getParent().substring(file.getParent().lastIndexOf("\\") + 1, file.getParent().length());
                if (dirName.equals("pos")) {
                    // Read each file in the "neg" category
                    train.posTermFreq = readFile(file, dirName, train.posTermFreq);
                } else if (dirName.equals("neg")) {
                    // Read each file in the "neg" category
                    train.negTermFreq = readFile(file, dirName, train.negTermFreq);
                }
            }
            if (file.isDirectory()) {
                // count = 0;
                //System.out.println("hit:" + file.getName());
                if (file.getName().equals("pos")) {
                    // Number of files in "pos" category
                    train.posLength = file.listFiles().length;
                } else if (file.getName().equals("neg")) {
                    // Number of files in "neg" category
                    train.negLength = file.listFiles().length;
                }

                // recursive call
                countFilesInDirectory(file, train, v);
            }
        }
        Object[] o = {train, v};
        return o;
    }

    /**
     * *************************************************************************
     * calculateVocabLength :
     *
     * @param posTermFreq : The list containing pos category terms with
     * frequencies
     * @param negTermFreq : The list containing neg category terms with
     * frequencies
     * @param v : vocabulary information
     * @return : The object of vocabData
     **************************************************************************
     */
    public static vocabData calculateVocabLength(Map<String, Double> posTermFreq, Map<String, Double> negTermFreq, vocabData v) {
        Set<String> uniqueVocab = new HashSet<>();
        double combinedfreq = 0.0;
        ArrayList<String> termToRemove = new ArrayList<String>();

        // remove terms that have combined frquency less than 5.
        // Add unique terms to vocab set.
        for (String s : posTermFreq.keySet()) {
            if (negTermFreq.keySet().contains(s)) {
                combinedfreq = posTermFreq.get(s) + negTermFreq.get(s);
                if (combinedfreq >= 5) {
                    uniqueVocab.add(s);
                } else {
                    termToRemove.add(s);
                }
            }
        }
        v.termSkip = termToRemove;
        v.vocabLength = uniqueVocab.size();
        return v;
    }

    /**
     * *************************************************************************
     * readFile :
     *
     * @param file : The file currently being read
     * @param dirName : The directory under which the file resides
     * @param termFreq : The map of the terms and frequencies for a particular
     * category
     * @return : The modified map of the terms and frequencies for a particular
     * category
     * @throws FileNotFoundException 
     **************************************************************************
     */
    public static Map<String, Double> readFile(File file, String dirName, Map<String, Double> termFreq) throws FileNotFoundException {
        Scanner trainFile = new Scanner(file);
        double freq = 0;
        try {
            while (trainFile.hasNextLine()) {
                if (trainFile.hasNext()) {
                    String term = trainFile.next();
                    // Skip special characters
                    Pattern p = Pattern.compile("[^A-Za-z]");

                    Matcher m = p.matcher(term);
                    boolean b = m.matches();
                    // If term is not a special character
                    if (b == false) {
                        if (termFreq.containsKey(term)) {
                            freq = termFreq.get(term);
                            freq++;
                            //System.out.println("frequency:" + freq);
                            termFreq.put(term, freq);
                        } else {
                            termFreq.put(term, 1.0);
                        }
                    }

                } else {
                    break;
                }
            }
        } catch (Exception e) {
            System.out.println("Exception:" + e);
        }

        trainFile.close();
        return termFreq;
    }

}
