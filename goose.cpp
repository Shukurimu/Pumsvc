/* Pneumonoultramicroscopicsilicovolcanoconiosis */
#include <stdio.h>
#include <cmath>
#include <string>
#include <vector>
#include <list>
#include <valarray>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#define ADAM_BETA1 0.9
#define ADAM_BETA2 0.999
#define ADAM_EPSILON 1.0e-8
#define PREDICTDAYS 5
#define TARGETINDEX 3
#define PRECISION 1.0e-2

inline double square(double&& x) {
	return x * x;
}

std::default_random_engine generator(std::random_device{}());
std::uniform_real_distribution<double> distribution(-0.05, 0.05);
void randomInitialization(std::valarray<double>& targetVector) {
	std::generate(std::begin(targetVector), std::end(targetVector), [&] () { return distribution(generator); });
	return;
}

struct ArgParse {
	double regularizer = 0.001;
	double adamlr = 0.001;
	int batchSize = 1024;
	int dimension = 128;/* individual stock embedding */
	int ompThread = 1;
	int history = 4;
	int featureLength = 5;
	int epoch = 500;
	std::string savePrefix = std::string("./goose_");
	std::string inputFile;
	std::string setting = "";
	
	ArgParse() {}
	~ArgParse() {}
	
	bool parseArgs(int argc, char** argv) {
		if (argc < 2) {
			printf(">>> The program needs at least 1 argument: InputFile\n");
			return false;
		}
		inputFile = std::string(argv[1]);
		if (parseOptions(argc, argv, 2)) {
			// omp_set_num_threads(ompThread);
			setting += "InputFile         : " + inputFile + "\n";
			setting += "EmbeddingDimension: " + std::to_string(dimension) + "\n";
			setting += "UseHistory        : " + std::to_string(history) + "\n";
			setting += "FeatureLength     : " + std::to_string(featureLength) + "\n";
			setting += "TrainingEpoches   : " + std::to_string(epoch) + "\n";
			setting += "BatchSize         : " + std::to_string(batchSize) + "\n";
			setting += "LearningRate      : " + std::to_string(adamlr) + "\n";
			setting += "Regularizer       : " + std::to_string(regularizer) + "\n";
			setting += "OpenmpThreads     : " + std::to_string(ompThread) + "\n";
			setting += "SavePrefix        : " + savePrefix + "\n";
			setting += "==================================================\n";
			printf("%s", setting.c_str());
			return true;
		}
		printf(">>> arguments error\n");
		return false;
	}
	
	bool parseOptions(int argc, char** argv, int optStart) {
		for (int i = optStart; i < argc; ++i) {
			if (std::string(argv[i]) == "--omp") {
				if (++i >= argc)	return false;
				int temp = std::stoi(std::string(argv[i]));
				ompThread = (temp < 1) ? ompThread : temp;
			} else if (std::string(argv[i]) == "-l") {
				if (++i >= argc)	return false;
				double temp = std::stod(std::string(argv[i]));
				adamlr = (temp < 0.0) ? adamlr : temp;
			} else if (std::string(argv[i]) == "-r") {
				if (++i >= argc)	return false;
				double temp = std::stod(std::string(argv[i]));
				regularizer = (temp < 0.0) ? regularizer : temp;
			} else if (std::string(argv[i]) == "-d") {
				if (++i >= argc)	return false;
				int temp = std::stoi(std::string(argv[i]));
				dimension = (temp < 1) ? dimension : temp;
			} else if (std::string(argv[i]) == "-h") {
				if (++i >= argc)	return false;
				int temp = std::stoi(std::string(argv[i]));
				history = (temp < 1) ? history : temp;
			} else if (std::string(argv[i]) == "-f") {
				if (++i >= argc)	return false;
				int temp = std::stoi(std::string(argv[i]));
				featureLength = (temp < 0) ? featureLength : temp;
			} else if (std::string(argv[i]) == "-e") {
				if (++i >= argc)	return false;
				int temp = std::stoi(std::string(argv[i]));
				epoch = (temp < 0) ? epoch : temp;
			} else if (std::string(argv[i]) == "-b") {
				if (++i >= argc)	return false;
				int temp = std::stoi(std::string(argv[i]));
				batchSize = (temp < 64) ? batchSize : temp;
			} else if (std::string(argv[i]) == "-s") {
				if (++i >= argc)	return false;
				savePrefix = std::string(argv[i]);
			} else {
				return false;
			}
		}
		return true;
	}
	
} args;

const double scoreWeight[] = { 0.0, 0.1, 0.15, 0.2, 0.25, 0.3 };

struct Stock {
	static int idCount;
	const int ID;/* index of row in ElemEmbedding */
	std::vector<std::valarray<double>> data;
	std::vector<double> truePrice;
	std::valarray<double> price1 = std::valarray<double>(0.0, args.featureLength);
	std::valarray<double> price2 = std::valarray<double>(0.0, args.featureLength);
	
	Stock(): ID(idCount++) {
		data.reserve(1600);
	}
	~Stock() {}
	
	void add(int& date, std::valarray<double>&& val) {
		price1 += val;
		price2 += val * val;
		data.push_back(val);
		return;
	}
	
	Stock* init() {
		double&& dataSizeDouble = static_cast<double>(data.size());
		price1 = price1 / dataSizeDouble;
		price2 = std::sqrt(price2 / dataSizeDouble - price1 * price1);
		/* get and remove testing data */
		for (size_t i = data.size() - PREDICTDAYS - 1; i < data.size(); ++i)
			truePrice.push_back(data[i][TARGETINDEX]);
		data.resize(data.size() - PREDICTDAYS);
		/* normalization */
		for (std::valarray<double>& v: data)
			v = (v - price1) / price2;
		return this;
	}
	
	double evaluate(FILE *sp, std::vector<double>& predPrice) {
		/* rescale */
		for (size_t i = 1; i <= PREDICTDAYS; ++i)
			predPrice[i] = std::round((predPrice[i] * price2[TARGETINDEX] + price1[TARGETINDEX]) * 100.0) / 100.0;
		
		double totalScore = 0.0;
		fprintf(sp, "p\t%.2lf\tt\t%.2lf\n", predPrice[0], truePrice[0]);
		for (size_t i = 1; i <= PREDICTDAYS; ++i) {
			double preddiff = predPrice[i] - predPrice[i - 1];
			int  predchange = (std::abs(preddiff) < PRECISION) ? 0 : ((preddiff > 0.0) ? +1 : -1);
			double truediff = truePrice[i] - truePrice[i - 1];
			int  truechange = (std::abs(truediff) < PRECISION) ? 0 : ((truediff > 0.0) ? +1 : -1);
			double&&  score = scoreWeight[i] * (((predchange == truechange) ? 0.5 : 0.0) +
				((truePrice[i] - std::abs(predPrice[i] - truePrice[i])) / truePrice[i]) * 0.5);
			totalScore += score;
			fprintf(sp, "%d\t%.2lf\t%d\t%.2lf\t%.2lf\n", predchange, predPrice[i], truechange, truePrice[i], score);
		}
		return totalScore;
	}
	
};
int Stock::idCount = 0;

struct ElemMatrix {
	const size_t nodeCount;/* set to 1 as scalar */
	const size_t dimension;
	std::vector<bool> dirty;
	std::vector<std::valarray<double>> gradients;
	std::vector<std::valarray<double>> adamBias1;
	std::vector<std::valarray<double>> adamBias2;
	std::vector<std::valarray<double>> embedding;
	int adam_t = 0;/* learning times */
	
	ElemMatrix(size_t c, size_t d): nodeCount(c), dimension(d) {
		dirty.assign(nodeCount, false);
		std::valarray<double> zeros(0.0, dimension);
		gradients.assign(nodeCount, zeros);
		adamBias1.assign(nodeCount, zeros);
		adamBias2.assign(nodeCount, zeros);
		embedding.assign(nodeCount, zeros);
		for (std::valarray<double>& x: embedding)
			randomInitialization(x);
	}
	
	void setGradient(std::valarray<double>& g, int entry = 0) {
		dirty[entry] = true;
		gradients[entry] += g;
		return;
	}
	
	/* when use entire embedding as W
	   the gradient of || Wx + b - y || ^ 2 is 2 * (Wx + b - y) * x.T
	   note that argument g = 2 * (Wx + b - y) */
	void setGradient(std::valarray<double>& g, std::valarray<double>& x) {
		std::fill(dirty.begin(), dirty.end(), true);
		std::transform(gradients.begin(), gradients.end(), std::begin(g), gradients.begin(),
			[&x] (std::valarray<double>& a, double& gi) { return a + x * gi; });
		return;
	}
	
	void update(int batchSize) {/* gradient descent */
		const double&& rho = args.adamlr * ((++adam_t > 30000) ? 1.0 :
			(std::sqrt(1.0 - std::pow(ADAM_BETA2, adam_t)) / (1.0 - std::pow(ADAM_BETA1, adam_t))));
		for (size_t i = 0; i < nodeCount; ++i) {
			if (args.regularizer > 0.0)
				gradients[i] += 2.0 * args.regularizer * embedding[i];
			else if (not dirty[i])
				continue;
			gradients[i] /= static_cast<double>(batchSize);
			adamBias1[i] = ADAM_BETA1 * adamBias1[i] + (1.0 - ADAM_BETA1) * gradients[i];
			adamBias2[i] = ADAM_BETA2 * adamBias2[i] + (1.0 - ADAM_BETA2) * gradients[i] * gradients[i];
			gradients[i] = 0.0;
			embedding[i] -= rho * (adamBias1[i] / (std::sqrt(adamBias2[i]) + ADAM_EPSILON));
		}
		std::fill(dirty.begin(), dirty.end(), false);
		return;
	}
	
	void dump(std::string fileName) {
		FILE *fp = fopen(fileName.c_str(), "w");
		for (std::valarray<double>& e: embedding) {
			for (double& v: e)
				fprintf(fp, "%.9e ", v);
			fprintf(fp, "\n");
		}
		fclose(fp);
		return;
	}
	
};

std::valarray<double> computeWxpb(/* Wx + b */
		std::vector<std::valarray<double>>& W,
		std::valarray<double>& x,
		std::valarray<double>& b) {
	std::valarray<double> result(b.size());
	std::transform(W.begin(), W.end(), std::begin(b), std::begin(result),
		[&x] (const std::valarray<double>& w, const double& bi) { return (w * x).sum() + bi; });
	return result;
}

int main(int argc, char** argv) {
	if (not args.parseArgs(argc, argv))
		return 0;
	
	std::unordered_map<std::string, Stock> stockMap;
	{
		char id[32];
		int date;
		double d1, d2, d3, d4, d5;
		FILE* fp = fopen(args.inputFile.c_str(), "r");
		while (fscanf(fp, "%s %d %lf %lf %lf %lf %lf\n", id, &date, &d1, &d2, &d3, &d4, &d5) == 7)
			stockMap[std::string(id)].add(date, std::valarray<double>({ d1, d2, d3, d4, d5 }));
		fclose(fp);
	}
	std::vector<std::pair<Stock*, int>> learningPair;
	for (auto& p: stockMap) {
		Stock* s = p.second.init();
		/* use [t - args.history, t - 1] to predict [t] */;
		for (int i = static_cast<int>(s->data.size()) - 1; i >= args.history; --i)
			learningPair.push_back(std::make_pair(s, i));
	}
	learningPair.shrink_to_fit();
	const int&& learningSize = static_cast<int>(learningPair.size());
	
	ElemMatrix stockFeature(Stock::idCount, args.dimension);
	ElemMatrix weightMatrix(args.featureLength, args.history * args.featureLength + args.dimension);
	ElemMatrix   biasVector(1, args.featureLength);
	
	const int vectorDimension = args.dimension + args.history * args.featureLength;
	std::slice embeddingSlice(0, args.dimension, 1);
	std::vector<std::slice> historySlice(args.history + 1);
	/* latest data starting with smaller index */
	for (int i = 1; i <= args.history; ++i)
		historySlice[i] = std::slice(args.dimension + (i - 1) * args.featureLength, args.featureLength, 1);
	
	double totalTrainingTime = 0.0;
	for (int e = 1; e <= args.epoch; ++e) {
		std::random_shuffle(learningPair.begin(), learningPair.end());
		const auto ttime1 = std::chrono::high_resolution_clock::now();
		double epochLoss = 0.0;
		
		// #pragma omp parallel for reduction(+ : epochLoss)
		for (int z = 0; z < learningSize; ++z) {
			Stock* currStock = learningPair[z].first;
			const int l = learningPair[z].second;
			
			std::valarray<double>& y = currStock->data[l];
			std::valarray<double>  x(vectorDimension);
			x[embeddingSlice] = stockFeature.embedding[currStock->ID];
			for (int i = 1; i <= args.history; ++i)
				x[historySlice[i]] = currStock->data[l - i];
			
			/* common gradient part */
			std::valarray<double> Wxpbmy2 =
				(computeWxpb(weightMatrix.embedding, x, biasVector.embedding[0]) - y) * 2.0;
			epochLoss += square(Wxpbmy2.sum() / 2.0);
			
			biasVector.setGradient(Wxpbmy2);
			weightMatrix.setGradient(Wxpbmy2, x);
			std::valarray<double> featureGradient(0.0, args.dimension);
			featureGradient = std::inner_product(
				weightMatrix.embedding.begin(), weightMatrix.embedding.end(),
				std::begin(Wxpbmy2), featureGradient,
				[] (std::valarray<double> a, std::valarray<double> b) { return a + b; },
				[&embeddingSlice] (std::valarray<double>& w, double& gi) {
					return std::valarray<double>(w[embeddingSlice]) *gi;
				});
			stockFeature.setGradient(featureGradient, currStock->ID);
			if ((z + 1) % args.batchSize == 0) {
				biasVector.update(args.batchSize);
				weightMatrix.update(args.batchSize);
				stockFeature.update(args.batchSize);
			}
		}
		const int&& finalBatchSize = learningSize % args.batchSize;
		if (finalBatchSize > 0) {
			biasVector.update(finalBatchSize);
			weightMatrix.update(finalBatchSize);
			stockFeature.update(finalBatchSize);
		}
		
		const auto ttime2 = std::chrono::high_resolution_clock::now();
		double&& dt = std::chrono::duration<double>(ttime2 - ttime1).count();
		totalTrainingTime += dt;
		printf("\rEpoch %3d/%d - %.3fs - Loss: %.4e", e, args.epoch, dt, epochLoss);
		fflush(stdout);
	}
	printf("\nTotalTrainingTime: %.3fs\n", totalTrainingTime);
	
	biasVector.dump(  args.savePrefix + "bias.txt");
	weightMatrix.dump(args.savePrefix + "weight.txt");
	stockFeature.dump(args.savePrefix + "stock.txt");
	FILE *fp = fopen((args.savePrefix + "mapping.txt").c_str(), "w");
	for (auto& kv: stockMap)
		fprintf(fp, "%s %d\n", kv.first.c_str(), kv.second.ID);
	fclose(fp);
	
	/* predict */
	FILE* sp = fopen((args.savePrefix + "score.txt").c_str(), "w");
	fprintf(sp, "%s", args.setting.c_str());
	double totalScore = 0.0;
	for (auto& kv: stockMap) {
		fprintf(sp, "\n%s\n", kv.first.c_str());
		Stock* const currStock = &kv.second;
		const int l = currStock->data.size();
		std::list<std::valarray<double>> historyInput;
		for (int i = 1; i <= args.history; ++i)
			historyInput.push_back(currStock->data[l - i]);
		
		std::vector<double> predPrice({ currStock->truePrice[0] });
		for (int p = 0; p < PREDICTDAYS; ++p) {
			std::valarray<double> x(vectorDimension);
			x[embeddingSlice] = stockFeature.embedding[currStock->ID];
			size_t i = 0;
			for (std::valarray<double>& vec: historyInput)
				x[historySlice[++i]] = vec;
			std::valarray<double> yp = computeWxpb(weightMatrix.embedding, x, biasVector.embedding[0]);
			historyInput.pop_back();
			historyInput.push_front(yp);
			predPrice.push_back(yp[TARGETINDEX]);
		}
		totalScore += currStock->evaluate(sp, predPrice);
	}
	fprintf(sp, "\nTotalScore: %lf\n", totalScore);
	fclose(sp);
	printf("\nTotalScore: %lf\n", totalScore);
	return 0;
}
