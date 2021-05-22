CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��"��`B      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N-�   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �u   max       >%      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�
=p��   max       @F�\(�     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ə����    max       @vnfffff     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q@           l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�           �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �]/   max       >j~�      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��=   max       B4zv      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�w�   max       B4zB      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�s�   max       C�P�      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C�X�      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N-�   max       P�|�      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��PH�   max       ?�'�/�V�      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �u   max       >�      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>޸Q�   max       @F�\(�     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ə����    max       @vnfffff     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @Q@           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�-�qv   max       ?�&���     �  M�         	            
                        #      Y         u   6      �               %         �      1             <   �   .   
   	                     
   {      F      QN�VMNµ�N�.'N/P�O�*dN�hN�lZN�M'N��$O��N��KN�5N�zNh�jP ��N�נP��N2��O���P��	O�Y�N�ETPb��O��hO6��O_�@N� O �NPb<N���PK��Nk��O��@OͰ�N��N-�O�JP�O<�4N�)yN��]N?5#O�0 O���O���NԂN�
�N�(QO��
Nah�O�H�Og�Oc�u�q����/�T���D���#�
�t�;o;�`B;�`B<D��<D��<T��<T��<u<�C�<�t�<���<�9X<�9X<�j<���<�/<�/<�`B=�w='�=,1=49X=49X=8Q�=@�=H�9=T��=T��=T��=T��=Y�=]/=aG�=q��=�C�=�\)=��=��P=��P=���=���=��T=���=�S�=�S�>%������	���������}{|������������}}}}��������������������7/7<GHMPH<7777777777xvw���������������x#)/2<=AB@</#������������������������������ #/2<FHRUUULH</$#  ������������������������������������������������������"/;=HPHGA;0/"��������������������_XZaz�����������zmf_)+5755)��)BSgleUNB5 ����`\Vafnxvna``````````�������������������������'+1>FF8)����CBF[hmt�������t[OBC��������������������#N[t��������t[N3&�������������������������������������������������������������������������������������������������������� 

����������#'/:<?C<7/.$#��������
�������"")/351/"����������

��������)463/)$������!%)/1)���������������������������������������������������!��������
#%/134.#
����������	


��������������������������
	
!#'#








��)?INOLB6))������BFB>6%))����)+/,)(%��&$#$&)*56;=>==952,)&��������������������������������������������������
���������
��������������������������������������������

����������˽��ĽнԽݽ��ݽнƽĽ������������������n�{ŇŔŗřŔōŋŇ�{�v�q�n�m�h�n�n�n�n���*�4�6�C�E�C�>�6�*����������EEEE!EEEED�EEEEEEEEEEE���(�5�L�b�l�m�i�`�N�A�8�(�"���
����(�5�A�I�N�S�N�D�A�5�(��������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������������������������������������������������������������������������a�m���������z�m�T�H�/�"�����"�/�H�a����������������������������������þ����������������ûùôøù�������������������T�a�l�m�s�v�y�z���z�p�m�a�]�T�S�N�M�T�T����������������������������������������������������{�q�n�k�n�s���Z�f�k�j�g�f�]�Z�M�M�E�K�M�W�Z�Z�Z�Z�Z�Z��T�e�i�g�V�;�"�	�������������������������������������������������������������āčĚĥĬĳĵĸĵķĦĚč�u�i�l�j�t�|ā���
�<�sŊřśŇ�b�I�0�������ĿĻ�������4�@�]�b�Y�S�@�4�����ܻлܻ�����4�A�M�Z�c�f�g�f�^�Z�M�A�7�4�/�0�4�4�4�4��)�\�c�c�^�O�6� �����������������������;�M�Y���v�f�M�@�'�����ܻ޻�����*�-�6�9�;�6�5�*�����������������(�A�N�]�g�p�p�g�`�N�A�5�(���������������ĿпѿܿѿĿ�������������������ÇÓàìù������ùìàÙÓÇ�|�z�x�yÅÇ�;�H�Q�J�H�?�;�/�*�/�1�1�;�;�;�;�;�;�;�;���	�����	�������������������������Çãêèôú����ìÓ�z�a�N�L�M�R�a�p�nÇ��#�0�<�=�<�;�0�#������������g¦������¿¦�t�[�N�5�,�$�)�4�H�[�g�)�1�8�:�/�������ݿҿҿؿ������)�������ĿοĿ����������������������������Ľнݽݽ�ݽнĽ����ĽĽĽĽĽĽĽĽĽĽ����(�4�C�K�I�A�8�(�����������������!�:�I�R�R�I�:�-���ɺĺ��úֺ�m�{�����������{�y�m�`�T�E�?�G�I�T�`�e�m���	���"�.�2�.�-�"� ��	��������������Ϲܹ޹�����߹ܹϹʹĹʹϹϹϹϹϹ�ǔǡǭǯǯǭǬǡǛǔǓǔǔǔǔǔǔǔǔǔ�ʾ׾���	������׾ʾ����������������ʺ?�>�A�:�I�Y�e�r�~�����������~�r�h�Y�E�?����������������������ƳƧƚƎƕƚƤƳ���O�\�h�uƁƄƁ�w�u�h�\�O�C�6�*�!�*�6�A�O�/�<�H�U�Z�]�U�H�<�/�)�-�/�/�/�/�/�/�/�/����������������������������������������D{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DnDkDmD{������������������������������������޼��'�4�=�>�7�-�'����ܻԻܻ������лӻлͻ˻Ļ���������������������������E�E�E�E�E�E�E�E�E�EsEoEnEqEuEzE�E�E�E�E� V H } B _  @  ; # @ � n i  Z ; ` 8 I >   }  n v M [ T B * a " d [ ) (  p W E a 1 4 } 1 + # D H [ (    �  �  �  8  *    �    !  $    �  E  �  �  �  �  :  M  �    �  �    {  N  �  s  �  �  z  {  �  �  �  1  �  �  �  �  �  W  �  �    x  �  �  /  �  B  ]  �Y��]/��t��o<e`B:�o:�o<o<�h<���<�<ě�<ě�<�o=H�9<ě�=��<�1=<j>O�=��-=8Q�>j~�=]/=Y�=ix�=8Q�=���=D��=P�`>H�9=L��=ȴ9=���=m�h=ix�=�`B>G�=���=�o=�7L=��=�j=���=���=�{=���=�1>P�`=�Q�>8Q�>I�>S��B'�B
�OB&DB��BW�B-UB$�B��B�Bz-B�B�A��B4zvB �%BiB�:B�oB�RB~rBn�B��B	�B!��B��B��BvB"Z�B&B;�B��A��=B��B�Bn4B  �B"IsB"f BL|BF�B�'B*�B$B~B�XB��B{;B-
�B�5B �B�B_�B\�B?�B
� B?�B�yB��B?�B@B�wB�	B��B�B8oA�w�B4zBB �	B�)B�5B�#B��B>@BAZB��B	>�B!��B�B=BAB"7B��B=eB��A��nB�B�CBJ0B >�B"B�B"@}B@BHfB�:B9�B1GB��B͎BFSBD�B-!LB�HB?�B��B�ZB�A(�A���A���C�f:A���A�.�C�P�A���AҲ�A���A��<A���A���AV�;A�U}A? _A�p^A�p�A޻3A�g@Ȩ�A<{kA�o@��CA�(#A�x2Aw7�A˂XA�g�A���A�t�A���A���A��Au��A(�A4 �@[��Aj7FA\r�>�s�BϕAR	@D�B�BujA�#qA"xEC���A�C@1@��C��A(y	A�ӪA���C�g�A�G�A�o�C�X�A�xEAҌWA��WAϒ}A΁hA���AW�xA���A?�A�H,A���Aނ(A��@�rA=
�A�y/@��A��sA�oAwA�w�A��4A��A��A��nA��DA�r�At��A'�A4��@[L�Aj��A\/>��B��AP�@��B�NBXAÂ�A#�C��A�$&@�!E@�j�C�"         
                                    #      Y         u   7      �               &         �      2             =   �   /   
   	                   	   
   |      F      R               #               %               %      =         9   '      /   )                     +      )   !            )               #   %                                          !                              #      7         5         !                              )   !                                                         N�VMN�C�N�.'N/P�O�4,N�hN�lZN�M'Ne�Oų<Ny�N�5N��Nh�jP`vN�נP�|�N2��O�� Py�N��N�ETOԘ5O`b�O6��O_�@N� N�	NPb<N���OlT~Nk��O��@OͰ�N��N-�Os�O�٭O<�4N�)yN��]N?5#O�̵O��O���NԂN�
�N�(QO1Nah�OUjN�Ok�  a  �  )  �  �  Z    �  `  �  �  �    u  �  (  x  �  �  
�  3       �  �    x  ;  c  �  G  r  S     �  Y  �  �  
V  �  B  �  �  "  �  �  �  B  �  Z  �  �  �u�m�h��/�T����`B�#�
�t�;o<e`B<#�
<�t�<D��<u<T��<�t�<�C�<�`B<���<�j=\)=T��<���=��=o<�`B=�w='�=H�9=49X=49X=��=@�=H�9=T��=T��=T��=�o=�
==]/=aG�=q��=�C�=�hs=��w=��P=��P=���=���>o=���=��m=�l�>�������	����������~~�������������������������������������7/7<GHMPH<7777777777�~}�����������������#)/2<=AB@</#������������������������������+/0<HKOHE<1/++++++++������������������������������������������������������"/;B>;/+"��������������������^\[]az����������zmc^)+5755)����);`gaUNB5"	���`\Vafnxvna``````````������������������������$&,9AA5)���NMOOZ[fhsrohg\[[SONN��������������������=768?N[gt������tg[F=�������������������������������������������������������������������������������������������������������� 

����������#'/:<?C<7/.$#��������������������"")/351/"����������

��������)463/)$������!%)/1)��������������������������������������������������������������
#%/134.#
����������	


��������������������������
	
!#'#








���)>HMOKB6)������)2@>:6������)+/,)(%��&$#$&)*56;=>==952,)&�������������������������������������������������������������
����������������������������������������������

�������Ž��ĽнԽݽ��ݽнƽĽ������������������n�{ŇŏŔŗŔŊŇŇ�{�{�s�o�n�n�n�n�n�n���*�4�6�C�E�C�>�6�*����������EEEE!EEEED�EEEEEEEEEEE���(�5�B�]�f�g�d�c�Z�N�A�0�%�$������(�5�A�I�N�S�N�D�A�5�(��������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�����������������������������������������������������������������������������H�T�a�m�~����y�m�a�H�/�"�����"�/�H����������������������������������������������������ûùôøù�������������������T�a�j�m�q�t�t�m�a�_�V�T�P�P�T�T�T�T�T�T��������������������������������������������������������u�n�q�����Z�f�k�j�g�f�]�Z�M�M�E�K�M�W�Z�Z�Z�Z�Z�Z�	�/�H�T�`�c�`�T�"�	�������������������	����������������������������������������āčĚĤīĳĵķĴĲĦĚč�v�i�m�k�t�}ā�
�<�c�}ŒŔŏ�{�b�I�0����������������
��'�3�4�@�A�@�@�4�'�����������4�A�M�Z�c�f�g�f�^�Z�M�A�7�4�/�0�4�4�4�4����)�6�G�P�R�M�D�6�)��������������������4�@�L�K�@�'��������������*�-�6�9�;�6�5�*�����������������(�A�N�]�g�p�p�g�`�N�A�5�(���������������ĿпѿܿѿĿ�������������������Óàìù����ùììàÓÇ�~�}ÇÏÓÓÓÓ�;�H�Q�J�H�?�;�/�*�/�1�1�;�;�;�;�;�;�;�;���	�����	��������������������������n�zÇÓààèëêàßÓÇ�z�m�c�b�d�l�n��#�0�<�=�<�;�0�#������������g¦������¿¦�t�[�N�5�,�$�)�4�H�[�g�)�1�8�:�/�������ݿҿҿؿ������)�������ĿοĿ����������������������������Ľнݽݽ�ݽнĽ����ĽĽĽĽĽĽĽĽĽľ���(�4�<�E�C�A�4�(�����������������1�:�<�:�.�!�������غԺغ��m�{�����������{�y�m�`�T�E�?�G�I�T�`�e�m���	���"�.�2�.�-�"� ��	��������������Ϲܹ޹�����߹ܹϹʹĹʹϹϹϹϹϹ�ǔǡǭǯǯǭǬǡǛǔǓǔǔǔǔǔǔǔǔǔ�׾���� ���׾ʾ������������������þ׺e�r�~�����������������~�r�m�Y�W�L�B�R�e����������������������ƳƧƚƎƕƚƤƳ���O�\�h�uƁƄƁ�w�u�h�\�O�C�6�*�!�*�6�A�O�/�<�H�U�Z�]�U�H�<�/�)�-�/�/�/�/�/�/�/�/����������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D~D�D�D�D�D�������������������������������������޼���'�4�6�9�2�(�������������������û˻ʻûû�������������������������E�E�E�E�E�E�E�E�E�E�E�E�EuEtEpEoEsEuE~E� V E } B d  @  & % 2 � [ i  Z 5 ` 7 I 6   Q  n v V [ T  * a " d [    p W E ^ 4 4 } 1 + ! D 5 M *    �  �  �  8  �    �    r  �  �  �  �  �  m  �  >  :  <  3  �  �  �  �  {  N  �  �  �  �  �  {  �  �  �  1  �  -  �  �  �  W  �  R    x  �  �    �  �    �  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  a  ^  \  [  Z  Z  Z  \  ]  L  9  $    �  �  �  �  �  e  ?  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  )  $       �  �  �  �  �  p  d  Q  4    �  �  �  �  �  t  �  �  �  �  �  �  �  s  f  [  Q  G  6     	  �  �  �  �  �  I  f  x  �  �  t  j  V  ;  (  H  L  0    �  �  �  �  �  }  Z  T  M  E  <  3  '      �  �  �  �  �  q  M  *    �  �       �  �  �  �  �  �  i  J  *    �  �  �  x  >  �  �  f  �  �  �  �  �  �  �  �  �  �  �  {  r  f  W  H  7       �      !  (  A  [  `  S  ;    �  �  �  W    �  ~  &  �  o  j  w    �  �  |  t  l  a  T  D  0    �  �  �  �  R     �  t  �  �  �  �  �  �  �  �  �  �  �  }  g  P  9     �  �  "  �  �  �  �    2  4  /  '  %           �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  (  �  �  [    �  d  u  r  p  n  k  i  f  a  Z  S  L  F  ?  8  0  )  !      
  �  �  �  �  �  �  �  y  Z  =  !  �  �  �  �  R    �  w    (  %  "        �  �  �  �  �  �  m  N  -    �  �  �  |  W  l  x  t  _  F  2    �  �  �  l  9  �  �  :  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  �  �  z  b  C    �  �  �  D    �  t    �  ^  
  �  
  ^  
�  
�  
�  
�  
�  
�  
Y  
  	�  	?  �  �  Q    �  �  �  �    �  3  V  ^  ^  X  V  `  �  �  !  2  %  �  �  q    �  �  �   O         �  �  �  �  �  w  N    �  �  ]    �  �  1  �  [  �  �  �  �    �  �    �  �  Z  �  j  �  �  i  �  
�  :  �  N  �  �  �  �  �  �  �  �  g  M  @  4    �  �  c    �  �  �  �  �  �  �  �  t  [  @  $    �  �  �    �  "  �    z    �       �  �  �  �  �  h  >    �  �  z  b  %    �  �  x  p  h  a  Y  O  =  +      �  �  �  �  �  �  �  �  �  u  �  "  $  7  6  &    �  �  �  d    �  F  �  1  �  �  =    c  Y  P  F  =  2  &         �  �  �  �  �  |  f  O  9  "  �  �  �  �  �  �  �  �  �  �  �  }  j  W  K  I  F  F  F  F  
R  
�  �  �  �  +  �  �  4  G  B  ,  �  y  �  �  L  
�  �    r  l  g  a  \  V  Q  E  6  '    
  �  �  �  �  �  �  �  ~  S  H  9  (    �  �  �  e  F  �  �  ~  T    �  Z  �  u  �     �  �  �  �  �  �  {  r  b  Q  D  -    �  �  f    7  d  �  �  �  �  �  y  s  o  l  j  f  b  ]  U  G  9    �  g    Y  S  N  H  D  E  G  H  D  7  +      �  �  �  �  �  �  v  �  {  �  �  �  �  �  �  a  :    �  l  	  �  
  r  �  �  y  �  n  �  =  }  �  �  �  �  �  z  0  �  @  �  
�  
  d  �  C  
V  
D  
*  
  	�  	�  	i  	&  �  �  ?  �  t  �  h  �    H  �  k  �  �  �  �  �  }  j  \  Q  K  H  E  A  <  6  /  (  %  <  R  B  *    �  �  �  �  �  �  �  p  O  .  	  �  �  �  �  d  H  �  �  �  �  t  \  D  -    �  �  �  �  �  q  T  5     �   �  �  �  �  �  r  \  >    �  �  �  Y  $  �  �  t  7    �  0        "        �  �  �  �  T    �  �  &  �  p  8  X  �  �  �  �  v  e  Q  5    �  �  �  F  �  �  8  �  \  �  �  �  �  �  �  �  j  P  6  "    �  �  �  �  �  �  �  �  �  |  �  �  �  �  }  \  ;    )  @  D  6  )  #         !  !     B  8  -    	  �  �  �  �  �  �  u  e  U  D  2      �  �  �  �  y  
  c  �  �  �  �  �  s  	  �  �  �  �  �  	�  c  �  Z  W  T  O  G  @  2       �  �  �  �  �  �  �  �    n  ]  O  �  �  �  �  �  �  �  N    
�  
o  	�  	w  �    2  R  2  �  v  �  �  �  n  >  	  �  �  5  �  �  /  �  �  ?  �  �  {  :  �  �  �  �  �  |  ]  1  �  �  S  �  a  �    
  �  3     