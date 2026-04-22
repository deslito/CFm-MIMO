# Project Assignment: Cell-Free Massive MIMO

## 1 Introduction to Cell-free Massive MIMO

Current mobile networks are implemented based on a cellular architecture where a centrally located BS serves all UEs within its coverage area. The coverage area is determined by the propagation environment. Since the received signal power decays with distance, multiple distributed BS' are deployed over the geographical area and each UE chooses the one that provides the strongest signal among them. The consequence of this is that UEs experience non-uniform coverage since some UEs are located close to the BS while others are at the cell boundary. Additionally, UEs at the cell edge are exposed to inter-cell interference from neighbouring BS, which lowers the Signal to Interference and Noise Ratio (SINR).

Massive MIMO [1] is one of the key technologies deployed in 5G due to its ability to enhance the SE and EE [1], [2]. It uses very large antenna arrays to serve multiple users simultaneously using the same time-frequency resources. As the number of antennas grows large, the effective channel appears deterministic and the individual channel vectors between the UEs and BS become almost orthogonal, thereby enabling the use of simple linear processing.

CF-mMIMO is a distributed implementation of mMIMO. Here, rather than deploying a macro BS with many antennas, a large number of small, low power APs, each consisting of a few antennas are distributed randomly over a large geographical area. The total number of antennas is still significantly larger than the number of UEs and all APs are connected to one or more central processing units (CPUs). All APs cooperate jointly to serve all the UEs via fronthaul links [3]. In this way, in addition to all the benefits of massive MIMO, it also provides uniform coverage since each UE is close to at least one AP. Additionally, the network is not partitioned into cells but rather each UE is served by a cluster of APs which may overlap for different UEs hence the cell-free architecture.

In the original CF-mMIMO [3], all APs were used to simultaneously serve all the users which is impractical because it would increase network complexity and the signalling load on the fronthaul when the number of UEs is large. Moreover, due to propagation effects, not all APs can effectively serve a UE. This motivated the idea of user-centric CF-mMIMO in which only a small set of APs is selected to serve each UE [4] as shown in Fig. 1.

![Figure 1: Illustration of user-centric CF-mMIMO](.\images\figure1.png)
*Figure 1: Illustration of user-centric CF-mMIMO*

### 1.1 System Model

A CF-mMIMO system with \(M\) APs equipped with \(L\) antennas each and \(K\) single-antenna UEs is considered, where \(\mathcal{M} = \{1,2,\ldots ,M\}\) and \(\mathcal{K}\in \{1,2,\dots ,K\}\) is the set of APs and UEs respectively. Each UE selects a set of APs \(\mathcal{M}_k\subset \mathcal{M}\) to serve it, while each AP serves a set of \(\mathcal{K}_m\subset \mathcal{K}\) UEs. A group of UE-AP association matrices \(\mathbf{A}_{mk} = a_{mk}\mathbf{I}_L\) is defined with \(a_{mk} = 1\) if AP \(m\) serves UE \(k\) and zero otherwise.

### 1.2 Propagation Model

#### Path Loss and Shadowing

The average path loss (PL) at some distance, \(d\), from the transmitter is commonly modelled using a log-distance model and is typically given by

\[\overline{\beta} (d) = \overline{\beta} (d_0) + 10n\log \left(\frac{d}{d_0}\right), \quad (1)\]

where \(d_0\) is some reference distance at which the average path loss is known, and \(n\) is the path loss exponent which is equal to 2 in free space and can be larger depending on the propagation environment.

Additionally, there exists a varying amount of clutter between a transmitter and any two receivers located at two distinct locations but equal distance from the transmitter. To account for this, 1 has to be modified. It has been found that the overall path loss is a log-normal distributed random variable i.e.

\[\beta (d) [\mathrm{dB}] = \overline{\beta} (d) + \mathcal{X}_{\sigma_{st}}, \quad (2)\]

### 1.3 Uncorrelated Rayleigh Fading Model

A common channel model used in classical CF-mMIMO literature [3] is the well known NLOS uncorrelated Rayleigh fading model where the channel response between each single antenna UE \(k\) and the AP \(m\) is modelled as

\[\mathbf{h}_{mk}\sim \mathcal{CN}(\mathbf{0}_L,\beta_{mk}\mathbf{I}_L), \quad (3)\]

where \(L\) is the number of antennas per AP and \(\mathbf{I}_L\) is the \(L\times L\) identity matrix. The Gaussian distribution models the small scale fading while the variance \(\beta_{mk}\) accounts for the large scale fading variations including path loss and shadowing.

### 1.4 Power Allocation

We use a heuristic power allocation scheme [5]. Each AP selects the power allocated to each UE proportional to its channel gain. The motivation is that since a UE is served by multiple APs, it is more intuitive to allocate more downlink power from APs with strong channels in order to maximize the SNR. Similarly, each AP should prefer to transmit to those UEs to which it has a good channel to minimize interference. The proposed power allocation algorithm is given by

\[p_{mk} = \left\{ \begin{array}{ll}P_{\mathrm{max,dl}} = \frac{(\beta_{mk})^v}{\sum_{k'\in\mathcal{K}_m}(\beta_{mk})^v} & k\in \mathcal{K}_m\\ 0 & k\notin \mathcal{K}_m \end{array} \right., \quad (4)\]

where \(p_{mk}\) is the power that AP \(m\) allocates to UE \(k\), \(P_{\mathrm{max,dl}}\) is the maximum transmit power per AP and the denominator is a normalization factor hence the APs always transmit with maximum power. The value selected for the exponent \(v\in [0,1]\) determines the specific power allocation and is typically selected as \(v = 0.5\).

### 1.5 Block Fading Model

In mMIMO analysis, a block fading model is assumed whereby time-frequency resources are divided into coherence blocks with time and frequency intervals corresponding to the coherence time and coherence bandwidth respectively. Within a coherence block, the channel between any two antennas is time-invariant and frequency-flat and can therefore be described by a complex valued scalar gain. Furthermore, the channel realizations are assumed to be independent between different blocks giving rise to the block-fading model.

If \(B_{c}\) is the coherence bandwidth and \(T_{c}\) is the coherence time, then from the Nyquist sampling theorem, a signal that fits into a coherence block is uniquely described by \(\tau_{c} = B_{c}T_{c}\) complex valued samples which make up the transmission symbols.

### 1.6 Time Division Duplexing (TDD) Protocol

In TDD, the uplink and downlink channels are reciprocal implying that the forward channel matrix is simply a transpose of the reverse one; therefore the channel information can be acquired by sending only uplink pilots [1]. In the TDD protocol, \(\tau_{c}\) samples are allocated between \(\tau_{p}\) pilot signals, \(\tau_{d}\) downlink data and \(\tau_{u}\) uplink data such that \(\tau_{c} = \tau_{p} + \tau_{u} + \tau_{d}\), as shown in Fig. 2.

![Figure 2: Coherence block in TDD mMIMO protocol](.\images\figure2.png)
*Figure 2: Coherence block in TDD mMIMO protocol*

### 1.7 Uplink Pilot Transmission and Channel Estimation

CF-mMIMO can be operated using distributed or centralized processing. In centralized processing, the APs act as relays to forward all the pilot and data signals to the CPU and then channel estimation and data detection is done at the CPU. In the distributed case, only the data signals are sent to/from the CPU while the APs carry out channel estimation and select the precoding/decoding vectors locally. We will use distributed processing.

In the ideal case, the pilot sequences assigned to all UEs are orthogonal to each other. However, the coherence interval has a finite number of samples \(\tau_{c}\ll K\) which are shared among \(\tau_{p}\) pilots and \(\tau_{d} + \tau_{u}\) data. Therefore, there can only be \(\tau_{p}\) mutually orthogonal pilot sequences where \(K\gg \tau_{p}\). It is assumed that the network utilizes the pilot sequences denoted by \(\pmb {\phi}_1,\dots ,\pmb {\phi}_{\tau_p}\in \mathbb{C}^{\tau_p}\) which are chosen to satisfy \(||\pmb {\phi}_t||^{2} = \tau_p\). Each UE \(k\) transmits a pilot sequence \(\pmb{\phi}_{t_k}\) scaled by the square root of the uplink transmit power \(\sqrt{p_k}\). The received pilot signal at AP \(m\) is \(\mathbf{Y}_m^p\in \mathbb{C}^{L\times \tau_p}\) given by [3]

\[\mathbf{Y}_m^p = \sum_{k = 1}^K\sqrt{p_k}\mathbf{h}_{mk}\pmb{\phi}_{t_k}^T + \mathbf{N}_m, \quad (5)\]

where \(\mathbf{N}_m\in \mathbb{C}^{L\times \tau_p}\) is the additive white Gaussian noise at the receiver whose elements are i.i.d as \(\mathcal{C}\mathcal{N}(0,\sigma_{\mathrm{ul}}^2)\) and \(\sigma_{\mathrm{ul}}^2\) is the noise variance. In order to estimate the channel \(\mathbf{h}_{mk}\), each AP \(m\) correlates the received pilot signal with the normalized conjugate of UE \(k\)'s pilot \(\phi_{t_k}\) to give

\[\begin{array}{rl} & {\mathbf{y}_{mk}^{p} = \frac{1}{\sqrt{\tau_p}}\mathbf{Y}_{m}^{p}\phi_{t_k}^{*}}\\ & {\qquad = \sum_{j = 1}^{K}\sqrt{\frac{p_j}{\tau_p}}\mathbf{h}_{mj}\phi_{t_j}\phi_{t_k}^{*} + \frac{1}{\sqrt{\tau_p}}\mathbf{N}_m\phi_{t_k}^{*}}\\ & {\qquad = \underbrace{\sqrt{p_k\tau_p}\mathbf{h}_{mk}}_{\mathrm{Desired~term}} + \underbrace{\sum_{j\in\mathcal{P}_k\backslash\{k\}}\sqrt{p_j\tau_p}\mathbf{h}_{mj}}_{\mathrm{Interference}} + \underbrace{\mathbf{n}_{t_k m}}_{\mathrm{Noise}}} \end{array} \quad (6)\]

where \(\mathbf{n}_{t_k m}\) is equivalent to the last term in the second line of 6 and \(\mathcal{P}_k = \{j: t_j = t_k, j = 1, \dots , K\} \subset \mathcal{K}\) is the set of UEs which use the same pilot as UE \(k\) and \(t_j = \{1, \dots , \tau_p\}\). The first and third terms of 6 contain a scaled version of the desired channel and noise respectively. The second term gives rise to the well-known phenomenon called pilot contamination resulting from the UEs which share UE \(k\)'s pilot. The minimum mean square error (MMSE) estimate of \(\mathbf{h}_{mk}\) based on \(\mathbf{y}_{mk}^{p}\) is then [6]

\[\hat{\mathbf{h}}_{mk} = \sqrt{p_k\tau_p}\mathbf{R}_{mk}\Psi_{t_k m}^{-1}\mathbf{y}_{mk}^p, \quad (7)\]

where \(\Psi_{t_k m}\) is the correlation matrix of \(\mathbf{y}_{mk}^{p}\) given by [6]

\[\Psi_{t_k m} = \mathbb{E}\left\{\mathbf{y}_{mk}^{p}\left(\mathbf{y}_{mk}^{p}\right)^{H}\right\} = \sum_{j\in \mathcal{P}_k}\tau_p p_j\mathbf{R}_{mj} + \sigma_{\mathrm{ul}}^2\mathbf{I}_L. \quad (8)\]

The estimate \(\hat{\mathbf{h}}_{mk}\) and the estimation error \(\tilde{\mathbf{h}}_{mk} = \mathbf{h}_{mk} - \hat{\mathbf{h}}_{mk}\) are independent random variables distributed as \(\hat{\mathbf{h}}_{mk} \sim \mathcal{CN}(\mathbf{0}_L, \mathbf{B}_{mk})\) and \(\tilde{\mathbf{h}}_{mk} \sim \mathcal{CN}(\mathbf{0}_E, \mathbf{C}_{mk})\) respectively where \(\mathbf{C}_{mk} = \mathbf{R}_{mk} - p_k \tau_p \mathbf{R}_{mk} \Psi_{t_k m}^{- 1} \mathbf{R}_{mk}\) is the error correlation matrix and \(\mathbf{B}_{mk} = \mathbf{R}_{mk} - \mathbf{C}_{mk} = p_k \tau_p \mathbf{R}_{mk} \Psi_{t_k m}^{- 1} \mathbf{R}_{mk}\). Since \(\mathbf{R}_{mk}\) and \(\Psi_{t_k m}^{- 1}\) depend on the channel statistics which are deterministic and change much slower than the channel vectors [6], the product \(\sqrt{p_k \tau_p} \mathbf{R}_{mk} \Psi_{t_k m}^{- 1}\) can be pre-computed and made available at the APs or CPU whenever required via the fronthaul [6].

The estimation accuracy is reduced by the presence of other users that share the pilot for UE \(k\). From 7, the channel estimates of two UEs \(j\) and \(k\) utilizing the same pilot are related by

\[\hat{\mathbf{h}}_{mj} = \sqrt{\frac{p_j}{p_k}}\mathbf{R}_{mj}\mathbf{R}_{mk}^{-1}\hat{\mathbf{h}}_{mk} \quad (9)\]

and are therefore correlated. If uncorrelated Rayleigh fading is used with \(\mathbf{R}_{mk} = \beta_{mk} \mathbf{I}_L\), then the channels between any two UEs that share the same pilot are simply scaled versions of each other.

### 1.8 Downlink Data Transmission

In the downlink, the CPU encodes the data signals \(q_{k}\) and sends them to the respective APs. Each AP then selects the precoding vectors locally based on the channel estimates and transmits \(\mathbf{x}_{m}\in \mathbb{C}^{L}\) given by

\[\mathbf{x}_m = \sum_{k = 1}^K\mathbf{A}_{mk}\mathbf{w}_{mk}q_k, \quad (10)\]

where \(\mathbf{w}_{mk}\in \mathbb{C}^{L}\) is the transmit precoding vector that AP \(m\) selects for UE \(k\) and \(\mathbb{E}\{|q_k|^2\} = 1\) is the signal meant for UE \(k\). Then the received signal at UE \(k\) is given by

\[\begin{array}{rl} & y_k^{\mathrm{dl}} = \sum_{m = 1}^{M}\mathbf{h}_{mk}^H\mathbf{x}_m + n_k\\ & \qquad = \sum_{m = 1}^{M}\mathbf{h}_{mk}^H\left(\sum_{k = 1}^{K}\mathbf{A}_{mk}\mathbf{w}_{mk}q_k\right) + n_k\\ & \qquad = \underbrace{\sum_{m = 1}^{M}\mathbf{h}_{mk}^H\mathbf{A}_{mk}\mathbf{w}_{mk}q_k}_{\mathrm{desired~signal}} + \underbrace{\sum_{j = 1}^{K}\sum_{m = 1}^{M}\mathbf{h}_{mk}^H\mathbf{A}_{mj}\mathbf{w}_{mj}q_j}_{\mathrm{inter~user~interference}} + \underbrace{n_k}_{\mathrm{noise}}, \end{array} \quad (11)\]

where \(n_k\sim \mathcal{CN}(0,\sigma_{\mathrm{dl}}^2)\) is the noise at the receiver. The downlink precoding vectors are selected based on the uplink combining ones as

\[\mathbf{w}_{mk} = \sqrt{p_{mk}}\frac{\mathbf{v}_{mk}}{\sqrt{\mathbb{E}\{||\mathbf{v}_{mk}||^2\}}}, \quad (12)\]

where \(\mathbb{E}\{||\mathbf{w}_{mk}||^2\} = p_{mk}\), which is the transmit power that AP \(m\) uses for UE \(k\), and is chosen in such a way that the per-AP power constraints are met.

### 1.9 Spectral Efficiency

The downlink SE has been shown to be [6]

\[SE_{k} = \frac{\tau_{d}}{\tau_{c}}\log_{2}(1 + SINR_{k})\mathrm{bit / s / Hz}, \quad (13)\]

where the effective SINR is given by

\[\begin{array}{rl} & {S I N R_{k} = \frac{\left|\sum_{m = 1}^{M}\mathbb{E}\left\{\mathbf{h}_{mk}^{H}\mathbf{A}_{mk}\mathbf{w}_{mk}\right\}\right|^{2}}{\sum_{j = 1}^{K}\mathbb{E}\left\{\left|\sum_{m = 1}^{M}\mathbf{h}_{mk}^{H}\mathbf{A}_{mj}\mathbf{w}_{mj}\right|^{2}\right\} -\left|\sum_{m = 1}^{M}\mathbb{E}\left\{\mathbf{h}_{mk}^{H}\mathbf{A}_{mk}\mathbf{w}_{mk}\right\}\right|^{2} + \sigma_{\mathrm{dl}}^{2}}} \end{array} \quad (14)\]

## 2 Assignment

The goal of this project is to implement two algorithms for AP-UE association i.e. you will be finding the binary association matrices to input in the SINR equation. For this assignment we shall use single-antenna APs so, \(\mathbf{A}_{mk}\) reduces to a scalar binary value \(a_{mk}\). You have been provided with starter code that implements a CF-mMIMO system with an AP-UE association scheme that uses the LSF gain. Details of this scheme are here [7].

### 2.1 Part One

Design an algorithm using k-means clustering.

You will choose appropriate features and cluster size to group users into clusters. After grouping users into clusters, assign APs to each cluster group. You should think of a good way to assign APs to each group to improve the overall system performance.

#### Reporting

Include in your report:

1. A brief description of your algorithm
2. Justification for your choice of features and cluster size
3. Compare the performance of your algorithm with the LSF-based one provided in terms of the SumSE and SE-per user. Note: to do this you need to run many random realizations and plot a CDF curve. The more realizations you run, the smoother your curve but also consider your computing power.

### 2.2 Part Two

Design an algorithm using TD3 [8], [9]. You have been provided with starter code that implements TD3 and fairly good hyper-parameters have already been chosen for you.

Our goal is to maximize the sum SE. Your task is to choose appropriate training tuples (s, a, r, s') to train your algorithm.

**Note:** This code has been written with python 3.11.5. If you are using a newer version of python, you may need to fix some errors. I recommend you instead work with a virtual environment (this is easy to do with conda) where you can install a lower version of python for this task so that you do not spend too much time trouble shooting errors.

#### Reporting

1. Explain how the TD3 algorithm works for this task from your understanding
2. Justification for your choice of states, actions and rewards
3. Compare the performance of your algorithm with the LSF-based one and your K-means algorithm

### 2.3 Housekeeping

- You will do this assignment in your groups of three
- Assignment due date April 24 or before
- You can use AI to brainstorm your ideas or improve your writing but you CANNOT use AI to fully do the work for you. You also shouldn't simply post the assignment and/or provided code in the AI tool and ask it to write your code or do it for you. This will be considered malpractice and you will not get a grade.
- Highlight where you have used AI truthfully in your report. Failure to declare your use of AI will also be considered malpractice.
- If time allows, you will present your assignment to the class. Otherwise, when you complete your assignment, you should submit your code, report and set up a mini "oral-exam" with me. This is to verify that you did the work yourself and that everyone participated in the assignment.

## References

[1] T. L. Marzetta, "Noncooperative cellular wireless with unlimited numbers of base station antennas," IEEE Transactions on Wireless Communications, vol. 9, no. 11, pp. 3590-3600, 2010. DOI: 10.1109/TWC.2010.092810.091092

[2] H. Q. Ngo, E. G. Larsson, and T. L. Marzetta, "Energy and spectral efficiency of very large multiuser MIMO systems," IEEE Transactions on Communications, vol. 61, no. 4, pp. 1436-1449, 2013. DOI: 10.1109/TCOMM.2013.020413.110848

[3] H. Q. Ngo et al., "Cell-free massive MIMO versus small cells," IEEE Transactions on Wireless Communications, vol. 16, no. 3, pp. 1834-1850, 2017. DOI: 10.1109/TWC.2017.2655515

[4] S. Buzzi and C. D'Andrea, "Cell-free massive MIMO: User-centric approach," IEEE Wireless Communications Letters, vol. 6, no. 6, pp. 706-709, 2017. DOI: 10.1109/LWC.2017.2734893

[5] G. Interdonato, P. Frenger, and E. G. Larsson, "Scalability aspects of cell-free massive MIMO," in ICC 2019 - 2019 IEEE International Conference on Communications (ICC), IEEE, 2019. DOI: 10.1109/icc.2019.8761828

[6] O. T. Demir, E. Bjornson, and L. Sanguinetti, "Foundations of user-centric cell-free massive MIMO," Foundations and Trends in Signal Processing, vol. 14, no. 3-4, pp. 162-472, 2021, ISSN: 1932-8346. DOI: 10.1561/2000000109

[7] H. Q. Ngo et al., "On the total energy efficiency of cell-free massive MIMO," IEEE Transactions on Green Communications and Networks, vol. 2, no. 1, pp. 25-39, 2018. DOI: 10.1109/TGCN.2017.2770215

[8] S. Fujimoto, H. Hoof, D. Meger, "Addressing function approximation error in actor-critic methods," in 35th International Conference on Machine Learning, vol. 80, 2018, pp. 1587-1596

[9] OpenAI, Twin Delayed DDPG, https://spinningup.openai.com/en/latest/algorithms/td3.html, accessed 2024-12-09