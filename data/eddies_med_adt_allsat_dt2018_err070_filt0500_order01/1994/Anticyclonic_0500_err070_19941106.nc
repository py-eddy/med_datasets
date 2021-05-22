CDF       
      obs    3   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�?|�hs      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P��"      �  x   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��t�   max       >J      �  D   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @E�z�G�     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ٙ����    max       @vZ�Q�     �  (   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @Q@           h  0    effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @��@          �  0h   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       >� �      �  14   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B+�>      �  2    latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B+��      �  2�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >h�   max       C���      �  3�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >?�.   max       C��,      �  4d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         L      �  50   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  6�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P=7�      �  7�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����$   max       ?��C�\��      �  8`   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��t�   max       >$�/      �  9,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>޸Q�   max       @E������     �  9�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��fffff    max       @vZ�Q�     �  A�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @Q@           h  I�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�           �  JP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�      �  K   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Fs����   max       ?��Y��|�     �  K�   
   M      9   4   
     L   "      
         
      9               $         5                  7   7   e                           K   ~   ?   *         
   	   1      N��O�ܲNu�PLH�P��"N�gO�Q�Pb�OJB�O�\�N���O}t�NA�N�Np�/O�PqM��O*B?N�}NB�tO�u�N���O�,O�ZN���O�;N�̋O�ҳO�W�P)J�PL��O���NQ!�N��N��OO�}�N�O�N�rN��O��P(�O�_sO���N�ONW+O;�N��O���Nސ=N����t��ě���o;D��;��
;�`B;�`B<T��<e`B<u<u<u<�t�<���<���<���<��
<ě�<���<�/<�/<�/<�`B<�h<�h<�h<�h<�<�<��<��=+=C�=�P='�='�=0 �=8Q�=D��=P�`=P�`=Y�=Y�=aG�=q��=q��=�%=�7L=�\)=\>J���������������������������������������������������������)Bgs�����tg^YKB\XU���������������m\"/;>=;1/"����������  �������$#(;Ng���������tN5-$+(%(/<HTaaidaYUH<2/+3+5B[gt��������tg[N3fhhehiqot}~���}vthf��������������������T[gghtttqgc[TTTTTTTT����������������������������������������ont�������������{wvo����������������������������������������+&%),//0<HKOMHH<5/++��������������������4,-)6@hot|}|xtqmh[Q4���������������������������	��������������������������


	

�������
#&/11/(#
�����

���������a\]g����������tqhkha����
#/05:AE</#
�������������������������)BNbhibYSLB5)���������
�����OJU^annxnaZUOOOOOOOO
"),,,)) 



456;ABNRTTQNEB<54444SUY[it���������tj[NSwyz������zwwwwwwwwww������������������������

������������xz���������zxxxxxxxx������������������������P[hmk`XB)�LHHHKO[t�����zth[UQL�������������������������)-)���115<BIHDB51111111111��$)/10.)����	�������dz���������������smdsonsz���������~zssss����������������������������������� �����������������������ؾ��(�A�M�q�����s�f�Z�A�(���ؽ������������ĿȿĿ�������������������������²��������¿¦�g�U�F�=�B�N�[�^�����������������������N�5�����,�}���T�a�j�m�r�x�y�m�a�^�T�S�M�M�T�T�T�T�T�T�������ûлۻջлû��������x�w�y�t�|������)�6�O�]�f�h�[�O�)��������������������������������������������������`�m�}�����������y�m�`�K�G�A�;�<�D�P�O�`�������������Ŀݿ�ݿѿĿ����������������'�4�@�F�B�@�9�4�'�������������'�����������������������������������������������������������������������������������������������������������������������Ž�����������ݽĽ����|�y�}�����Ľݽ�������ùϹ۹ٹϹ͹ù����������������������������������������������������|�z��}�������������������������������������s�����H�T�a�e�f�a�T�K�H�D�H�H�H�H�H�H�H�H�H�H��������ʾ׾����׾���������s�g�Y�f�àìôùÿ��ùñìãàÖÙßàààààà��"�.�;�G�M�Q�J�G�;�.�"���	��	�����#�/�H�U�a�p�w�u�j�<�/�#��������"�.�;�?�F�;�;�.�$�"������ �"�"�"�"�T�`�m�y�~���������~�y�m�f�T�I�G�?�G�L�T�L�Y�e�r�v�}�r�e�Y�L�K�B�L�L�L�L�L�L�L�L�T�a�m�p�q�m�a�T�H�;�/�-�!����/�;�H�T���5�;�G�R�X�T�N�A�(������������������*�6�O�a�wƂƅ��u�\�*�����žź��ƁƎƚ��������������ƳƎ�u�l�g�e�hƁDoD�D�D�D�D�D�D�D�D�D�D�D{DpDoDeDcDdDgDo��)�.�)�(��������������������Ŀſпѿݿ�ݿѿĿ����������������������(�)�3�(������������������Z�g�s�v���������������s�Z�P�K�F�A�6�N�Z��������������������������������������������������������Ҽ�������������������������������������������������������������������������������ùܹ������ܹϹù����������������û����'�4�B�L�f�Y�@�4�������ܻٻػ鼋�������ʼԼڼϼü���������y�p�i�r������������ĽϽѽ����������y�k�]�c�k�q�|���#�0�<�I�O�O�I�C�<�<�1�0�/�*�#����#�#�<�I�M�U�Z�U�I�<�9�3�<�<�<�<�<�<�<�<�<�<�������	��"�)�-�&�"� ��	��������������e�r�~�����������~�r�f�e�`�d�e�e�e�e�e�e����������#�,�+����������������������޻�����"�#�������������������E7ECEPETE\EeEgE`E\EPEDECE8E7E.E+E7E7E7E7 0 I 1 : K V 3   $ M } / Q U ? \ � I > 8 : ( ) # B & @ 8 , y M   r j O C D  v � R A ) . H Q / " m # B    �  j  �  �  �    7  �  �  �      L  Q  w    Z  �    ]  8  �  8  �  �  D  �    �  �  �    y  �  �  H  .  0  Y  �  �  �  A  �  6  9  �  �  	  �  �o=�\);D��=q��=ix�<�o=+>� �=D��='�<ě�=#�
<�9X<�`B<�j=���<���=t�=o<��=}�=�w=49X=��T=t�=D��=t�='�=ix�=� �=�{>1'=,1='�=@�=}�=D��=��=P�`=aG�>%>7K�=�=ě�=���=�%=��=���=�F=���>��BM>B"�~B��BvBv�A���B"��B	�B$B	)�B+B �B	rB��B<�B�B��B2>B��B�DBB�B!u�B3B&�BG�BcB#�B
-~B|B��BO1B�B5'B(eB�B	�B�BW,B#ĲBB�$B��B�2B+�>BL{B�{Bw/B��B�>B�BޑB@B"��B�_BI�B?�A���B"�{B	=hB:nB	@)BB�B!@>B	7oB�OB?�B��B��B@[B�Be:B>1B!�B?�B&fB@:BA\B#ޠB
��B�B�fB��B�+B<qBC�B�(B
�B��B?�B#W�B�
BWaB�TB�BB+��B?HB�B$KBAB>hB�PB��A��A7�YAuͭA�XA�A�m�@�R�A�;AҺhAi�2Av�&@��A�;ZA��A���A&�>lK�A�!�A��dA��fAK}�A�eLA`�Aè$A`|PAj�?���A�D,A��A���B�C���A�b�AxΫA���A��:A�lAѓ�@�kGA�\�>h�@�@�%A�KA��\A�RUA��?� �A҈�@�
�C���A�h�A8{�Au�A��SA���A�|�@��AԙMAҀAl�^At0r@�qA�KA���A��|A&��>@\A���A�H�A��!AIm4A�}�Aa��A�lEAa�Aj�?��A�#?A� /B 8�B�KC�� A�}Ax�A�+NA�buA�>�A�f/@��%A��k>?�.@��9@�AY�A��A���A�~�@+�AҀ�@� dC��,      M      9   5   
     L   #      
         
      :               $         5                  8   7   e   	                        L      ?   *            	   2            )      -   9      !   /      !                  #               '         !                  -   /                                 -      #               %                  )   )                                                                           '   -                                 #                           N��O_�rNu�PRP0N�{�OfO�O��HN���O�TN���O}t�NA�N�Np�/O�7M��NQ�IN�}NB�tO���N���O8{O��N���O�;NW�OrN�O���O���P=7�O+�NQ!�N��N��OO�}�N�O�N�rN��O�ZOځ�OB�oO�*RN�.;NW+O;�N��Os��Nސ=N��  z  �    1  K  �    E  $  H  8  B    �  �  k  u  �  ?  �  �  �    �  {  �  �  :  �  �  �  �  �  ?    �    �  �  �  �  �  	�  }  �  �  �    	�     ���t�<�j��o<e`B<�j<t�<#�
>$�/<�`B<���<u<u<�t�<���<���=,1<��
<�h<���<�/=�w<�/<�h=+<�h<�h<�<��=+='�=C�=}�=C�=�P='�='�=0 �=8Q�=D��=P�`=q��=�9X=�O�=�o=}�=q��=�%=�7L=���=\>J������������������������������������������������������������)5Bgs{���tg[NB5|{{��������������� "/;;;:/,"��������������������=:=DN[gt������tg[ND=5008<@HPUWUQH<555555;7;BN[gt}������tk[N;fhhehiqot}~���}vthf��������������������T[gghtttqgc[TTTTTTTT����������������������������������������������������������������������������������������������������+&%),//0<HKOMHH<5/++��������������������A<=<@EO[hmtuuqkh[OHA���������������������������
	��������������������������


	

�������
#&/11/(#
�����

	���������b^gt���������tsmimjb����
#/268?A</#
����������������������)5N^fg`XNIB5)��������

�����OJU^annxnaZUOOOOOOOO
"),,,)) 



456;ABNRTTQNEB<54444SUY[it���������tj[NSwyz������zwwwwwwwwww������������������������

������������xz���������zxxxxxxxx����������������������)BO[cd[OB)�NLMORY[ht}����th[RON������������������������������115<BIHDB51111111111��$)/10.)����	���������������������������sonsz���������~zssss����������������������������������� �����������������������ؾ(�4�A�M�Y�c�[�M�A�4�(����������(���������ĿȿĿ�������������������������²¿������½²¦�n�a�]�_�h�g�n�g�����������������������g�N�5�'�&�-�H�g�T�a�g�m�o�t�m�m�m�a�W�T�P�O�T�T�T�T�T�T�����������ûлѻû������������{�|�x�������)�6�@�G�I�G�@�6�)�������������������������������������������������`�m�y�~�����������y�m�`�T�G�A�A�G�V�T�`�������������Ŀݿ�ݿѿĿ����������������'�4�@�F�B�@�9�4�'�������������'�����������������������������������������������������������������������������������������������������������������������Ž����Ľнݽ����ݽнĽ��������������������ùϹ۹ٹϹ͹ù������������������������������������������������������������������������������������������������s�����H�T�a�e�f�a�T�K�H�D�H�H�H�H�H�H�H�H�H�H�����������ʾվھݾ׾ʾ�������x�q�w���àìôùÿ��ùñìãàÖÙßàààààà��"�.�;�G�L�P�I�G�;�.�"���	������/�H�U�a�n�u�t�h�Q�<�/�#����
���#�/�"�.�;�?�F�;�;�.�$�"������ �"�"�"�"�T�`�m�y�~���������~�y�m�f�T�I�G�?�G�L�T�L�Y�e�r�t�y�r�e�Y�M�L�E�L�L�L�L�L�L�L�L�T�a�m�o�p�m�a�W�H�;�/�"����"�/�;�H�T���5�8�E�M�R�I�A�5�(���������� �����*�6�S�k�x�z�u�h�\�O�*�������������ƁƚƳ��������������Ƴƀ�u�o�i�g�iƁD{D�D�D�D�D�D�D�D�D�D�D�D�D~D{DnDmDoDqD{��)�.�)�(��������������������Ŀſпѿݿ�ݿѿĿ����������������������(�)�3�(������������������Z�g�s�v���������������s�Z�P�K�F�A�6�N�Z��������������������������������������������������������Ҽ�������������������������������������������������������������������������������Ϲܹ�������Ϲ������������������ùϼ���'�6�B�I�Q�M�C�4�'��������������������ʼ̼˼ż������������|�~���������������ýȽ������������y�q�l�f�k�y�����#�0�<�I�M�M�I�@�<�0�0�/�#�!��!�#�#�#�#�<�I�M�U�Z�U�I�<�9�3�<�<�<�<�<�<�<�<�<�<�������	��"�)�-�&�"� ��	��������������e�r�~�����������~�r�f�e�`�d�e�e�e�e�e�e����������&����������������������޻�����"�#�������������������E7ECEPETE\EeEgE`E\EPEDECE8E7E.E+E7E7E7E7 0 : 1 ; X W .  # G } / Q U ? H � 4 > 8 4 ( ( % B & = 3 $ u M  r j O C D  v � M 9  3 . Q / " < # B    �  �  �  �  �  �  �  �  �  <      L  Q  w  S  Z  o    ]    �    �  �  D  p  �  R  �  y  d  y  �  �  H  .  0  Y  �  �  �  �  F  �  9  �  �  �  �    ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  z  z  x  n  b  N  9      �  �  �  l  7  �  �  �  c  �  �  �  �  �  F  y  �  �  �  �  �  _  /  �  �  G  �  �  �  �          
      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �       -  0  /  -  *  (      �  �  |  Q    �  �  -  �  �  �  �  &  4  9  F  K  @  *      �  �  �  �  W  �  �  #  d  p  |  �  �    w  l  ^  N  <  %    �  �  y  H    �  �          	  �  �  �  �  �  �  �  �  �  ~  E    �  �  ~  �    �  o  �  �  g  �  ?  >     l  �    �  �  �  �  4    �  /  m  �  �  �    !  $    	  �  �  �  W  �  �  �  �  �  �  (  C  G  B  6  "    �  �  �  �  �  �  �  �  Q  �  [  �  8    �  �  �    /  1  !    �  �  �  �  �  b  A    �  �  B  +    �  �  �  �  h  E  &    �  �  �  �  t  /  �  �  R              �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    R  Y  Z  `  h  m  p  r  s  p  h  Y  ;    �  :    �  �  �  �  �  �  �  �  �  �  �  �  �  |  q  e  J  )  	   �   �  �  @  �    =  T  b  j  k  c  U  >     �  �  �      �  �  u  y  }  �    h  Q  :       �  �  �  �  i  J  )    �  �  r  o  m  n  {  �  �  �  �  �  �  �  �  �  n  R  5    �  >  ?  3  '          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �           	  
    	    �  �  �  �  �  �  �  m  �  %  L  j  �  �  �  �  �  d  8  	  �  �    D  �  �  �  �  �  o  T  8    �  �  W    �  �  Y  p  u  E    �  �  r  9          �  �  �  �  �  �  l  I     �  �  p  (  �  �  J  �  �  �  �  b  +  �  �  �  �  �  �  B  �  �  ;  �  /  �  �  {  r  i  ^  P  B  5  '      �  �  �  �  �  �  }  e  L  2  �  �  �  �  �  m  O  .    �  �  w  ;  �  �  m    �  S  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  o  f    �  :  5  8  8  .  !    �  �  �  �  �  �  o  J  &    �  �  �  �  �  �  �  �  �  �  �  v  F    �  �  T  '  �  �  j  .  �  �    V  s  ~  }  n  M    �  �  6  �  �  h    �  �  �     �  �  �  �  �  �  �  �  �  l  E    �    '  �  �  +  �  �   �  �  �  D  y  �  �  �  �  p  (  �  5  �  �  �  �  
�  	�  �  �  �  �  �  �  �  �  �  �  t  i  Q  -    �  �  �  g  =     �  ?  4  (      
  
  
  
  
    �  �  �  �  �  �  c  5        �  �  �  �  �  �  �  �  w  b  N  7      �  �  �  �  �  �  �  �  �  �  �  �  i  >    �  �  Y    �  x  :  �  �      �  �  �  �  �  �  �  �  �  �  �  �    r  S  /    �  �  �  �  �  �  �  �  �  �  �  �  q  L  	  �  S  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  S  :  "  �  �  �  �  �  �  �  �  �  �  �  �  q  V  <        �   �   �  �  �  �  �  �  �  >  �  �  >  
�  
g  	�  	�  	  Z  |  x  n  �  I  �  F  �  �  �  �  �  ?  �  f  �  U  �  
�  	�  �  X  �  �  	�  	�  	�  	�  	�  	�  	�  	�  	{  	H  		  �  Q  �  T  �    �  X  K  L  ^  n  x  }  x  j  N  )     �  �  T    �  '  w  �  !  �  ]  }  �  �  �  }  n  Z  C  '     �  �  t  E      �  �  �  �  u  e  V  F  8  -  !        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  d  O  =  1    	  �  �  �  �  H  �    
                    &  -  1  5  :  >  D  O  Z  �  �  	#  	;  	�  	�  	c  	3  �  �  A  �  Z  �  ;  �  �  s  �  �       �  �  �  G    �  �  N    �  _    �  N  �  >  �    �  l  +  �  �  R    �  �  B  �  �  0  �  >  �  �  �  2   c