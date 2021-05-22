CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��1&�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P�3      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �   max       ;ě�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>^�Q�   max       @FW
=p��     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @vrfffff     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P�           �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ϊ        max       @�c@          �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��u   max       ;�o      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B1��      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�E   max       B1�1      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >I��   max       C�w9      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >E"a   max       C�xF      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          g      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          C      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P�H      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�ݗ�+j�   max       ?�c�	�      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �   max       ;ě�      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>^�Q�   max       @FW
=p��     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @vq��R     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P�           �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���          �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��u%F   max       ?�c�	�     �  V�                  5         	      
                     $            f   	                              )            #               6      M   ?      
      
      B                        	         	    N��N��Og�}Nj��NZ��P.�PKM���NsMzO)�Nu�O6��O��O�r�O��N���Oj|`O��O��ZN�zO^�P7WFN�1FNNN�z�N���N���O�uO��N{�XOt��O��P+gtN��SN��!P=JOmPDrOs��N���O"�)P�3Ne�;O��pP�BN��Ow�}Of<VNi�4No�O��O+��N��^N0˄NՐ�Op�&N���NI69NYײO^5Np)
N�.�O�H�;ě�;D��;o:�o��o��o��o��o��o�ě���`B�#�
��o��o��t���t���t����㼣�
��9X��j��j��j���ͼ�����/���C��C��t��t���w�#�
�'',1�,1�0 Ž0 Ž<j�D���D���H�9�P�`�]/�aG��e`B�ixս}󶽁%��%��%��o��7L�������㽛�㽩�罬1��E��Ƨ��^ht������the^^^^^^^^()6>A>6/)��������������������#/<<HUWULHF<8/#��������������������#DZbfallt|zlOBJ[ikt���������th[FDJ��������������������S[chqnih[YSPSSSSSSSS�����

��������~����������{~~~~~~~~./<?HUZagkkaUH<743/.������� ����������)6OY_[KH>A6=BO[htz���th[OD=;<=���������������������������������������������
	
	���������������������������������� ������������9BN[dgt���tg[XNJC?9uz������������zqpt{u#01230.020'# ������������������������������������

���������BCDNOU\hjptsnh\ZOKCB)5BLN[gt����yt[NB5#0:IUbmqph`N<0)#����������������������������� ���������gt������������{tmeag�����!&����������������������������������������������������������������������������������������������%&����������������������STYalmnnmka]VTSSSSSS�����������������������$0BV]O6�����RTaammsmaUTIRRRRRRRR���
#'# 
������<HUanzy���zraU:1028<HJQUanqnjigea^USOKHH���������������zwvz�"/;DHJNONIH;/""���������������������������������������� #0<AJORSMH<0# ##0<=IOQPLI<0)#"!#��
#)/-#
	��������������������������;HT^aaa`YTH@;:8;;;;;)35?BFEIGEB5)KNR[glig`[ONHJKKKKKKst{���������{vtsssslnqrz}~znjhjlllllllrt|����������tommnrr��������������������!),)��EN[t�����������tgZNE�Ŀ������ĿĿѿڿۿܿԿѿĿĿĿĿĿĿĿĺɺǺ����������ɺֺ����������ֺɺɾs�f�_�c�n�s�������������������������s�����������������������������������������0�'�%�-�0�0�=�>�>�?�=�:�0�0�0�0�0�0�0�0�_�F�!���!�-�F�_�v�x���������������x�_���ؾʾ�������������.�;�B�D�N�L�;���������������������������r�l�r�x�~�������������~�r�r�r�r�r�r�r�r�����������������	��"�'�)�'�"��	��������������������� �������������������������������������������#�%��������s�n�g�`�k�s���������������������������s������������������������������������������	���&�/�;�H�T�a�g�l�a�R�H�;�/�"��l�k�_�[�X�V�_�_�l�n�x�x�����}�x�l�l�l�l�f�Y�F�5�2�2�9�M�_�f�r�y�v�r�p�t�~�r�n�f�@�3�(�!�0�4�@�Y�e�~�������������r�Y�L�@�����ѿ��������������Ŀѿ����������	� �������	���"�/�2�6�/�"�����m�b�c�]�a�b�f�m�z�������������������z�m���	���3�L�Y�r�~�����������~�e�3�����������������������ûǻлջܻ޻ܻϻû��a�U�\�a�f�n�p�s�r�n�a�a�a�a�a�a�a�a�a�a�����������������������������������������g�e�`�c�g�s�|�����������������s�g�g�g�g�	�����ݾ������	��"�&�#�"���	�	�	�� ������������	��������	���|�s�k�h�l�s���������������������������`�^�\�`�m�y���������y�m�`�`�`�`�`�`�`�`àÓÌÇÂ�z�yÃÇÓàìù��������ùìà����������������������Q�Y�N�M�B�)����3�@�v�����ʼ���!�$�!�����ʼ���f�@�3ŭŪŠşŠŭŰŹ������������žŹŭŭŭŭ�ʼ¼����������Ǽʼּݼ�����ڼּʼʻx�l�_�S�:�4�7�E�S�l�x�����ͻٻʻû����x���޻��������'�4�@�;�1�'�������ŠŏņņŒŠŭűž����� �
��0�6�*��Ҿ����׾Ӿ׾����	��"�.�7�=�;�.�"���������ƶ���������������������������������C�9�9�6�*���������&�7�;�<�C�M�C�t�G�.���G�y�������нԽ���&�������t����������������������������������������D�D�D�D�D�D�D�D�EEE*E7EDEGE=E+EED�D�Ѿ��������������	��.�B�>�.�"�	����ѹ¹����������������ùϹܹ����ܹϹù����������������� �0�6�:�B�8�0�$����������������������������������������������B�>�8�9�B�O�Y�U�O�O�B�B�B�B�B�B�B�B�B�BŇŇŇŏŔşŠšŨŠŞŔŇŇŇŇŇŇŇŇ���l�g�a�g�l�y���������Ľѽܽڽн��������������������(�4�:�B�=�4�*�(������ѿ˿Ŀ¿ÿ��Ŀѿݿ������ݿѿѿѿ��=�5�3�=�I�V�V�]�V�I�=�=�=�=�=�=�=�=�=�=�����������(�.�1�(� �����������������������������������������	���g�a�Z�O�N�Z�g�i�s�w�������s�g�g�g�g�g�g����¿²²²¿��������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��#�����!�#�/�<�H�U�U�Z�]�U�H�<�/�#�#�����������������������������������������B�<�B�H�O�W�[�f�h�m�l�k�h�g�[�O�B�B�B�Bā�y�w�w�{�~āĆčĚĦĳĶĹĸĴĬĚčā j  , u x 8 _ Z < / J - " 7 B S > ( K j 3 ) Z N , - . � < D / O � E U N / g W I g S O > D Z ) ) D c 0 0 T O  w j t � < l ` X    �    �  �  �    �  @  �  x  �  �  H  �  !    �  �  )  �  �  8  �  H  �  �      +  �  �  n  �  �  !  [  �     <  ~  �  *  �  �  �  ,  �  �  �  S  e  m     :  �  �  �  �  �  0  �  �  E;�o���㻃o�D���ě��Y��u�D���D����o��o��1�+�+�\)��`B�C��]/����/�,1�   �o�+�o�C��t��@��e`B�0 Ž�%��7L���T�8Q�aG���\)���㽓t�����P�`�e`B����}�J����-��+�� Ž�t�����o��j���w������^5��vɽ��T��Q콼j��/���`�%��uBV�BNB!��B��B�B7kBʈB �[B\B3{B+�BmTB4�B��B`�B!5:B!��B"��B*�1A���B��B�JB%I�B"&�BlBN	B1��B	=B&��Bf�B+%B,oB-�BQ	B!BxB�mBB�By6B}�A���B�]B��A�G�Bk�B�SBBB�A��bB��B~B%�6B&�BB��A���Br�B�pB
nTB�QB
5�BcB��B	��BKB�B"�B�gB��BHvB?�B!3�BChB@@B?LB�YB7�BHrB��B!*|B!@�B"��B+5A�EB	?�B�!B%��B!�_B:�BM2B1�1B	J]B&�UBJ[B�nB8sB-�
BV�B!9sBB�B@KB�.BgqA�|�B�gBʈA���B>�B��B�B
��A���B.B@OB%�gB&>�B�0B��A���B0�B��B
?�B�OB
@,B�@B�aB	�lAy�|@=hAGɳA���B
a@���A[�A.N�@f�A�>A���A�{aA�dA�vhA�y�@��@��?��lA|�A���A��`?��i@��A�וAI(�A��gAZ"�AZ<A�%Al�A˷^A��@��lA���@��@��}@�	�A�[kAZ��BxA���A!=7A�<C�`OAX�>I��B	+�A�.�A���A�FA 
�A4��A{dyB:�A��jA�ܔA���A��C�w9A�HPA�DA�!HA�%�Ax��@<[jAẸA�}�B
E@��iAYCA.�@7�A�LwA� �A��VA�SA�qA�gO@� @���?��aAy�A��sA��w?�Ns@���Aƃ~AH)eA��[A[-A[OA�R�Ak4A�y[A��A+A�%@�7�@�Do@ß�A�vaA]w�Bw�A��HA#`A��C�U`AY k>E"aB��A�E�Aؐ>A�}A)sA5/Az��B}�A���A�u�A�h/A��KC�xFA��A���A��A޻�                  5         	      
                     $            g   
   	                           *            #            	   6      M   @                  B         	               	         
   !                  /   +                     #            #   '         +                     %         %   =         '      5            E         +                                                                        #                        #               '         !                     %         %   =         '      5            C                                                               N��N�Og�}Nj��NZ��O�o�O���M���NsMzO�N-�N�|O3[O�x�On��N���O'_3OC�*O��ZN�zO^�O��N�1FNNN�z�N���N���N�yUO��N{�XOt��O��P+gtN��SN��!P=JOF��PDrOs��N���N�JOP�HNe�;Oz�O�	�N��O\X�Of<VNi�4No�OI#OΗN��^N0˄NՐ�Op�&N���NI69N}O^5Np)
N�.�O�H�  �  �  �  ^  �  m    �  J  �  �  �  �  �  4  �  G    �  �  X  
�  �  �  �  :  �    �    5  [  I  d  Q  �  �  ,  �  }  �  �    �  �  �  �  �  p  �  	}  �    �  �  �  b  �  �  �  �  �  T;ě���o;o:�o��o�T�����
��o��o��`B�t��T����1��t����㼓t���1�o���
��9X��j�]/��j���ͼ�����/����P�C��t��t���w�#�
�'',1�<j�0 Ž0 Ž<j�H�9�L�ͽH�9��%�����aG��ixսixս}󶽁%���
��7L��o��7L�������㽛�㽩�罰 Ž�E��Ƨ��^ht������the^^^^^^^^)68=96+)��������������������#/<<HUWULHF<8/#��������������������)6BO[`bdljaOB)#N[bhs����|{{th[NKIN��������������������S[chqnih[YSPSSSSSSSS�����

����������������������������8<BHUWabfdaUHF<:8888��������������������)6OW][HF;;6	?BO[hty~��}th[OF@>=?������������������������������������������������ ����������������������������������� ������������9BN[dgt���tg[XNJC?9z}����������������{z#01230.020'# ������������������������������������

���������BCDNOU\hjptsnh\ZOKCBHN[gt����tg[NBHHHHHH#0:IUbmqph`N<0)#����������������������������� ���������gt������������{tmeag�����!&����������������������������������������������������������������������������������������������%&����������������������STYalmnnmka]VTSSSSSS�����������������������#/BLRO6�����RTaammsmaUTIRRRRRRRR�����

������<BDHUahmruskgaUH:78<HJQUanqnjigea^USOKHH{��������������{xxy{"/;DHJNONIH;/""����������������������������������������#*0<CHKKHB<0###.02<ILONJI<0+$#!!#��
#)/-#
	��������������������������;HT^aaa`YTH@;:8;;;;;)35?BFEIGEB5)KNR[glig`[ONHJKKKKKKst{���������{vtssssknz{}|znkikkkkkkkkkkrt|����������tommnrr��������������������!),)��EN[t�����������tgZNE�Ŀ������ĿĿѿڿۿܿԿѿĿĿĿĿĿĿĿĺֺѺɺ����Ǻɺֺ��������ֺֺֺ־s�f�_�c�n�s�������������������������s�����������������������������������������0�'�%�-�0�0�=�>�>�?�=�:�0�0�0�0�0�0�0�0�[�K�?�9�:�@�F�S�l�����������������x�l�[�����޾׾ܾ����	��.�;�<�F�C�;�.���������������������������r�l�r�x�~�������������~�r�r�r�r�r�r�r�r�����������������	��"�%�'�%�"��	�������������������������������������������������������������������������������s�q�n�o�s������������������������������������������������� ��������������������(�/�;�H�U�a�f�j�`�P�H�;�/�"��l�k�_�[�X�V�_�_�l�n�x�x�����}�x�l�l�l�l�K�@�:�8�7�@�B�M�Y�f�p�n�k�p�r�u�r�f�Y�K�L�K�?�9�@�L�Y�e�r�~�������������r�e�Y�L�����ѿ��������������Ŀѿ����������	� �������	���"�/�2�6�/�"�����m�b�c�]�a�b�f�m�z�������������������z�m�e�L�3�-�%�&�,�@�L�Y�r�~�������������~�e���������������������ûǻлջܻ޻ܻϻû��a�U�\�a�f�n�p�s�r�n�a�a�a�a�a�a�a�a�a�a�����������������������������������������g�e�`�c�g�s�|�����������������s�g�g�g�g�	�����ݾ������	��"�&�#�"���	�	��������������	������	���������������|�s�k�h�l�s���������������������������`�^�\�`�m�y���������y�m�`�`�`�`�`�`�`�`àÓÌÇÂ�z�yÃÇÓàìù��������ùìà����������������������Q�Y�N�M�B�)����3�@�v�����ʼ���!�$�!�����ʼ���f�@�3ŭŪŠşŠŭŰŹ������������žŹŭŭŭŭ�ʼ¼����������Ǽʼּݼ�����ڼּʼʻx�l�_�S�:�4�7�E�S�l�x�����ͻٻʻû����x���������������,�4�<�8�.�'�����ŠŏņņŒŠŭűž����� �
��0�6�*��Ҿ����׾Ӿ׾����	��"�.�7�=�;�.�"���������ƶ���������������������������������*�$������	���$�*�4�6�9�;�6�,�*�*�z�G�.���G�y�������Ľн��#�������z����������������������������������������D�D�D�D�D�D�D�D�D�EEE*E9EAE6E*EEED����׾ϾȾžʾ׾��	��"�-�2�.�"�����¹����������������ùϹܹ����ܹϹù��������������	����0�3�8�@�6�0�$���������������������������������������������B�>�8�9�B�O�Y�U�O�O�B�B�B�B�B�B�B�B�B�BŇŇŇŏŔşŠšŨŠŞŔŇŇŇŇŇŇŇŇ�y�q�l�j�p�y�������������Ƚý����������y��������������(�4�7�?�:�4�(�$����ѿ˿Ŀ¿ÿ��Ŀѿݿ������ݿѿѿѿ��=�5�3�=�I�V�V�]�V�I�=�=�=�=�=�=�=�=�=�=�����������(�.�1�(� �����������������������������������������	���g�a�Z�O�N�Z�g�i�s�w�������s�g�g�g�g�g�g����¿²²²¿��������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��#�����!�#�/�<�H�U�U�Z�]�U�H�<�/�#�#�����������������������������������������B�<�B�H�O�W�[�f�h�m�l�k�h�g�[�O�B�B�B�Bā�y�w�w�{�~āĆčĚĦĳĶĹĸĴĬĚčā j  , u x  H Z < ) E . + : A S C 9 K j 3  Z N , - . j < D / O � E U N * g W I H T O 9 9 Z # ) D c * ) T O  w j t N < l ` X    �  �  �  �  �  �  7  @  �  <  `    u  �  �    �  �  )  �  �  �  �  H  �  �    1  +  �  �  n  �  �  !  [  �     <  ~    �  �    5  ,  �  �  �  S  �  5     :  �  �  �  �  (  0  �  �  E  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  �  ~  y  s  n  h  c  ]  X  R  R  V  Z  ^  b  f  j  n  r  v  �  �  �  �  �  �  �  �  �  f  <    �  �  r  .  �  �  2    �  �  �  �  �  �  �  �  �  �  �  �  w  n  i  d  `  _  ]  \  ^  R  G  ;  0  &              �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  Q  3    �  �  �  �  x  W  5      :  Q  V  i  m  i  e  ]  E    �  y     �  �  F    �  �  �  �  �    �    	  �  �  �  �  �  �  �  �  |  g  I  6  @  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  i  ]  J  >  2  !    �  �  �  �  �  �  �  |  e  O  5    �  �  �  �  �  �  �  �  �  t  e  V  H  8  %    �  �  �  �  �  �  t  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  \  I  <  .  �  �  �  �  �  �  �  �  u  d  R  L  M  F  >  2    �  �  `  S  q  �  �  �  �  �  �  �  �  �  �  z  d  I  0    �  �  �  �  �  �  �  �  �  �  �  �  n  \  D  +      �  �  s  7    )  3  *  %  '  '      �  �  �  �  t  M  (    �  �  r    �  �  �  �  �  �  �  �  �  �  �  w  ]  @  #    �  �  L  �  1  <  C  F  G  E  @  :  0  (  "          �  �  S    �  �  �  �  �  �            �  �  �  �  P  	  �  :  �    �  {  f  M  Y  Y  T  T  T  N  C  3      �  �  �  n  A   �  �  �  �  �  �  y  m  b  W  L  @  5  !    �  �  �  �  �  c  X  Q  G  :  0  *      �  �  �  �  �  _  -  �  �  �  �    	"  	�  
-  
w  
�  
�  
�  
�  
y  
7  	�  	  	  s  �  �  (  ;  �    �  �  �  �  �  �  ~  o  ^  J  4    �  �  �  d  &  �  �  M  �  �  }  r  h  _  Y  R  P  P  K  B  9  8  7  ;  @  -  �  �  �  �  �  �  �  �  �  �  ~  l  V  :      �  �  �  g  ?    :  0  &        �  �  �  �  �  �    j  Y  K  =  ^  �  �  �  �  �  �  �  �  �  �  x  j  X  D  0       �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  h  I  )  	  �  �  �  d  I  4  �  �  �  �  �  �  �  v  h  U  B  4  5  6  )    �  �        �  �  �  �  �  �  �  �  �  �  ~  �  �  �  �  �    .  C  5  '      �  �  �  v  G    �  �  >  �  �  I  �  ]  �    [  Z  W  O  P  L  >  *    �  �  �  �  �  w  /  �  @  �  �  I  %  �  �  �  |  9  �  �  H  �  �  4  �  �  b  #  �    �  d  [  R  I  A  7  ,         �  �  �  �  �  �  �  �  �  �  Q  O  I  =  .      �  �  �  �  a  =        n  �  �  �  �  �  �  �  h  D    �  �  �  �  g  3  �  �  p    �      �  �  �  �  �  �  �  u  R  )  �  �  �  E  �  �  ?  �  7  �  �  ,        �  �  �  �  �  �  s  A      �  �  �    ]  �  �  �  f  B    �  �  �  �  {  U  &  �  �  y  5  �  �  �  �  }  j  V  C  .       �  �  �  �  �  i  A    �  �  �  o  B  �  �  �  �  �  �  �  �  �  �  �  �  b  I  1    �  �    E  �  �  �  �  �  �  g  $  �  y  $  �  �  �  �  �  _  �  a  �    �  �  �  �  �  �  �  �  {  i  W  E  3  #  �  �  �  �  �  8  �  �  �  �  h    �  *  �      
�  
  	�  	    �  w  �    _  {  �  �  �  �  �  �  �  <  �  [  �  J  �  �  ,  M  m  �  �  l  F    �  �  �  Z    �  �  f  %  �  �  )  �  R  �  �  �  �  �  �  �  �  x  W  2    �  �  �  a  B  (     �   �  �  �  �  �  �  u  \  ?    �  �  �  J  �  �  R  �  �    �  p  S  6    �  �  �  �  p  F    �  �  �  d  7  	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      �  �  	!  	T  	t  	}  	s  	W  	2  	
  �  �  V     �  �    �  �  �  w  �  �  �  |  k  Q  1    �  �  o  #  �  _  �  ~  �  9        �  �  �  �  �  �  }  w  q  j  Z  H  5  #        #  �  �  �  w  g  W  F  5  #    �  �  �  �  �  u  W  9    �  �  �  �  �  �  _  ;    �  �  �  S    �  �  L  �  g  �  >  �  �  �  �  �  �  �  �  �  �  \  5    �  �  �  W    ]   �  b  U  I  <  0  '        �  �  �  �  �  �  �  �  �  l  J  �  �  �    l  Z  K  =  /    �  �  �  �  z  Y  8     �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  m  b  V  J  �  �  �  �  �  l  Q  3    �  �  �  �  ?  �  �    i  �    �  �  �  �  �  �  �  v  k  _  S  H  :  *      �  �  �  �  �  z  Z  A  )    �  �  �  �  u  C    �  �  H    �  �  D  T  7    �  �  �  |  C    �  {  )  �  T  �  J  !  �  u  =