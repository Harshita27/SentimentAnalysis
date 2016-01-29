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
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;
import java.util.Set;

/**
 *
 * @author Harshita
 * Naive Bayes text categorization system to predict whether movie reviews are 
 * positive or negative. The data for this “sentiment analysis” task were first 
 * assembled and published in Bo Pang and Lillian Lee, 
 * “A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization
 * Based on Minimum Cuts”, Proceedings of the Association for Computational Linguistics, 2004.
 * 
 * The program implements a bag-of-words, naive Bayes classifier using add-1 
 * (Laplace) smoothing.
 */

/**
 * ***************************************************************************
 * This class's object holds all the necessary data as fetched from the 
 * model file.
 * ***************************************************************************
 */

class naiveBayesData
{
    double vocabLength;
    double posLength;
    double negLength;
    double posPrior;
    double negPrior;
    public Map<String, Double> posTermProb = new HashMap<String, Double>();
    public Map<String, Double> negTermProb = new HashMap<String, Double>();
    public Map<String, Double> calculatedPosProb = new HashMap<String, Double>();
    public Map<String, Double> calculatedNegProb = new HashMap<String, Double>();
}

/**
 * ***************************************************************************
 * Main Class.
 * ***************************************************************************
 */
public class nbtest {
    public static void main(String args[]) throws FileNotFoundException
    {
        // The model file as read
        File model_file = new File(args[0]);
                //"C:\\Users\\Harshita\\Documents\\NEU\\IR\\Assignment-6\\textcat\\ModelFile.txt");
        // The test directory as read
        File folder = new File(args[1]);
                //"C:\\Users\\Harshita\\Documents\\NEU\\IR\\Assignment-6\\textcat\\test");
        PrintWriter writer = new PrintWriter(folder+"results_test_final .csv");
        File[] listOfFiles = folder.listFiles(); 
        naiveBayesData nb = new naiveBayesData();
        
        /**
         * *******************************************************************
         * Method to read the model file.
         * *********************************************************************
         */
        naiveBayesData nb2 = readModel(model_file, nb);
       
         /**
         * *******************************************************************
         * Method to classify each of the documents in the test dirctory as per
         * Naive Bayes conditional probability model.
         * *********************************************************************
         */
        nb2= classifyDocuments(folder, nb2);
        
        /**
         * *******************************************************************
         * Method to calculate the ratio and print the top 20 terms.
         * *********************************************************************
         */
        calculateratioFor20Terms(nb2);
    
        /**
         * *******************************************************************
         * Write the predictions file.
         * *********************************************************************
         */
        for(String s: nb2.calculatedPosProb.keySet())
        {
            if(nb2.calculatedNegProb.keySet().contains(s))
            {
                  writer.print(s+",");
                  writer.print(nb2.calculatedPosProb.get(s)+",");
                  writer.print(nb2.calculatedNegProb.get(s));
            }
             writer.println();   
        }
          
        writer.close();
    }
    
    /***************************************************************************
     * calculateratioFor20Terms :
     * @param nb                : The object of class naiveBayesData that stores all the
     *                            information from the model file.
     * @throws FileNotFoundException 
     **************************************************************************/
    public static void calculateratioFor20Terms(naiveBayesData nb) throws FileNotFoundException
    {
       
        Map<String, Double> listofRatios = new HashMap<String, Double>();
        for(String s: nb.posTermProb.keySet())
        {
            if(nb.negTermProb.keySet().contains(s))
            {
                listofRatios.put(s, nb.posTermProb.get(s)/ nb.negTermProb.get(s));
            }
        }
        
       computeTop20(listofRatios, "PosToNeg");
       listofRatios.clear();
       for(String s: nb.posTermProb.keySet())
        {
            if(nb.negTermProb.keySet().contains(s))
            {
                listofRatios.put(s, nb.negTermProb.get(s)/ nb.posTermProb.get(s));
            }
        }
        computeTop20(listofRatios, "NegToPos");
        
  }
    
    /***************************************************************************
     * computeTop20         : Method to sort the list
     * @param listofRatios  : The list of ratios of neg to pos or pos to neg
     * @param ratio         : The category of the ratio
     * @throws FileNotFoundException 
     **************************************************************************/
    public static void computeTop20(Map<String, Double> listofRatios, String ratio) throws FileNotFoundException
    { PrintWriter writer = new PrintWriter("RatioOfTop20Terms"+ratio+".csv");
         Set<Entry<String, Double>> set = listofRatios.entrySet();
        List<Entry<String, Double>> list = new ArrayList<Entry<String, Double>>(set);
        Collections.sort( list, new Comparator<Map.Entry<String, Double>>()
        {
            public int compare( Map.Entry<String, Double> o1, Map.Entry<String, Double> o2 )
            {
                return (o2.getValue()).compareTo( o1.getValue() );
            }
        });
        int counter=0;
        for(Map.Entry<String, Double> entry:list){
            if(counter<20)
            writer.println(entry.getKey()+","+entry.getValue());
            counter++;
        }
        writer.close();
    }
    
    /***************************************************************************
     * readModel        :
     * @param file      : The model file containing training data
     * @param nb        : The object of naiveBayesData that holds all the information
     * @return          : The object of naiveBayesData that holds all the information     
     * @throws FileNotFoundException 
     **************************************************************************/
    public static naiveBayesData readModel(File file, naiveBayesData nb) throws FileNotFoundException
    {
       Scanner modelFile = new Scanner(file);
       String[] split;
       try {
            while (modelFile.hasNextLine()) {
                if (modelFile.hasNext()) {
                    String term = modelFile.next();
                    split=term.split(":");
                    if(split[0].equals("length(vocab)"))
                    {
                        nb.vocabLength=Double.parseDouble(split[1]);
                    }
                      if(split[0].equals("length(pos)"))
                    {
                        nb.posLength=Double.parseDouble(split[1]);
                    }
                        if(split[0].equals("length(pos)"))
                    {
                        nb.negLength=Double.parseDouble(split[1]);
                    }
                    if(split[0].equals("P(pos)"))
                    {
                        nb.posPrior=Double.parseDouble(split[1]);
                    }
                    if(split[0].equals("P(neg)"))
                    {
                        nb.negPrior=Double.parseDouble(split[1]);
                    }
                    if(split[0].equals("pos"))
                    {
                        String[] probabilities = split[1].split(",");
                       nb.posTermProb.put(probabilities[0], Double.parseDouble(probabilities[1]));
                    }
                    if(split[0].equals("neg"))
                    {
                        String[] probabilities = split[1].split(",");
                       nb.negTermProb.put(probabilities[0], Double.parseDouble(probabilities[1]));
                    }
                }
                else 
                    break;
            }
          }
          catch(Exception e)
          {
              System.out.println("Exception:"+ e); 
          }
       modelFile.close();
       return nb;
    }
    
    /***************************************************************************
     * classifyDocuments    :
     * @param directory     : The test directory containing documents to be classified
     * @param nb2           : The object of naiveBayesData that holds all the information
     * @return              : The object of naiveBayesData that holds all the information   
     * @throws FileNotFoundException 
     *************************************************************************/
    public static naiveBayesData classifyDocuments(File directory, naiveBayesData nb2) throws FileNotFoundException
    {
      //  naiveBayesData nb = new naiveBayesData();
        int posCount = 0;
        int negCount = 0;
        for (File file : directory.listFiles()) {
            //System.out.println("file:"+ file);
            if (file.isFile()) {
                String dirName = file.getParent().substring(file.getParent().lastIndexOf("\\") + 1, file.getParent().length());
                // Read the files contents
                nb2=readFile(file, dirName, nb2);    
            }
            if (file.isDirectory()) {
                // Recursive call
                classifyDocuments(file, nb2);
            }
        } 
        return nb2;
    }
    
    
    /***************************************************************************
     * readFile         :
     * @param file      : The file to be categorized
     * @param dirName   : The parent directory of the file
     * @param nb        : The object of naiveBayesData that holds all the information
     * @return          : The object of naiveBayesData that holds all the information
     * @throws FileNotFoundException 
     **************************************************************************/
    public static naiveBayesData readFile(File file, String dirName, naiveBayesData nb) throws FileNotFoundException
    {
       Scanner testFile = new Scanner(file);
       
        double freqPos = 0.0;
        double freqNeg = 0.0;
        double classPos=0.0;
        double classNeg=0.0;
        double condProbPos=0.0;
        double condProbNeg=0.0;
        try {
            while (testFile.hasNextLine()) {
                if (testFile.hasNext() ) {
                    String term = testFile.next();
                    if(nb.posTermProb.keySet().contains(term))
                    {freqPos=nb.posTermProb.get(term);}
                   // System.out.println("freqp:"+ freqPos);
                    if(nb.negTermProb.keySet().contains(term))
                    {freqNeg=nb.negTermProb.get(term);}
                     condProbPos += freqPos;
                     condProbNeg += freqNeg;
              } else {
                    break;
                }
            }
        } catch (Exception e) {
            System.out.println("Exception:" + e);
        }
        // Calculate the conditional probability of a term belonging to a particular class
        classPos = nb.posPrior + condProbPos;
        classNeg = nb.negPrior + condProbNeg;
        nb.calculatedPosProb.put(dirName+","+file.getName(), classPos);
        nb.calculatedNegProb.put(dirName+","+file.getName(), classNeg);
        testFile.close();
        return nb;   
    }
}
