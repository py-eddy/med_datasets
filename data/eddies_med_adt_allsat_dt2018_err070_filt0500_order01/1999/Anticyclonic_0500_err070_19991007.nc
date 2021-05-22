CDF       
      obs    3   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�I�^5      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�m   max       Ql�      �  x   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��1   max       =�G�      �  D   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�ffffg   max       @E��z�H     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���G�{    max       @vh��
=p     �  (   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @O�           h  0    effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�^�          �  0h   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��t�   max       >�%      �  14   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B,s�      �  2    latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�nb   max       B,@�      �  2�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =�'   max       C��      �  3�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =�4   max       C��      �  4d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         )      �  50   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          Q      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +      �  6�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�m   max       PO�      �  7�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��W���'   max       ?�?��      �  8`   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��1   max       >C��      �  9,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�ffffg   max       @E��Q�     �  9�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vh��
=p     �  A�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @O�           h  I�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��`          �  JP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         G    max         G       �  K   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��t�j~�   max       ?�n.��3     �  K�               *   3   �                        y               	      2   L         6            	   ;   ?      %      E   *      	      
      $  )   w   I               N/�N�ȘN>5$O_^�O�ÉPU'�P6�sNO>�O=+O|�NLGN��NE8�N��sQl�NF'�O]�JO:�PN"�0NT:�N���P�TP,�MN��sO�<O�w�M�mN)ȯN��5O�Pr�O�D{OuG�O�AyNb��P0_�OE�BO ��N�N�H�O{&yN���O]/�P9ƯOy�P/��N��/N�]O1�N�کN僱��1�u�T���o�o�o;��
;�`B<t�<D��<T��<�t�<�t�<�t�<��
<��
<�9X<ě�<���<�/<�/<�<��<��<��=+=\)=��=#�
=#�
=0 �=<j=@�=H�9=L��=]/=]/=}�=}�=�%=�+=�+=��w=��T=�1=� �=�E�=�j=�"�=�"�=�G�8ABNOQONEB8888888888��������

��������������������������"'/;ADEB@;2/"#/<HUanwnaUH</#�����#0=II?)������PLLNUan���������zn\Pmbgnz�{znmmmmmmmmmm�������������������������6BD?63 �������������������

�������
#)��������������������������/`kt��t[N���dbehotz}{tthdddddddd�����#$#������������������ ����)1+)��������������������  #.0<IQIG<60#      ^djt��������������m^�������
.1,%������spst�������������}ts��������������������vrrz|�������������}v������������������������

�����������T[_hmt����|tlhc[TTTT��������������������\cecit�����������tg\���������������,+,0?JO[hqolflkdYB6,���#/<HU\_UH</#��{{����������{{{{{{{{������+7<<2+���������

 !
��������������������������������������������������������������������������������������������������������z}���������������~{z���t[NB2--1BNgt�������������
����������*.( ���������nqsyz����������znnnn #$/;<?<://#����������������������������	���������������

 ������������������ŹŸŹź���������������������(�5�A�N�P�Z�^�Z�R�N�A�5�*�(�$�&�(�(�(�(�����������������������������������������T�a�i�m�w�z�����z�m�a�T�;�,�"� �.�;�G�T�����������������������������������������������û����l�F�>�4�:�S�_�x�����ùܹ��� ������Ϲù����a�\�d�s���������������������������������������������������������ԺѺ޺����������������ɼμмμƼ����������������������<�H�O�P�P�H�<�4�1�0�<�<�<�<�<�<�<�<�<�<�y�����������y�m�`�Y�`�d�m�x�y�y�y�y�y�y��!�-�2�.�-�!�������������ÓàìíùúùøìàÓÇÅÇÈÏÓÓÓÓƳ������;�=�)���ƧƁ�\�<�*���E�Y�zƳ�����	�����	��������������������������	�"�.�4�.�,�"��	������׾;׾����f�s���������������������s�f�Z�P�U�Z�f�Z�`�f�a�Z�M�K�K�M�T�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z���'�.�0�'����
�����������'�4�8�@�L�I�H�@�6�4�,�'�$�#�'�'�'�'�'�'���	��!�(�*�(�"���	�������������������5�B�[�g�t�~�t�h�[�B�5�������������5�A�M�Z�\�_�W�M�A�4�(������(�.�4�?�A�����������������������������Ľݽ�ù����������������������ùìá×ØÞìù�zÇÈÇ�|�z�n�n�h�j�n�t�z�z�z�z�z�z�z�z�/�;�H�J�L�H�;�/�+�,�/�/�/�/�/�/�/�/�/�/�@�D�L�U�Y�^�Y�Y�L�H�@�6�3�2�3�9�@�@�@�@�:�F�\�l���������x�u�l�_�S�D�:�-�)�-�3�:�y�����ݿ���� ����꿫�����z�w�u�s�y�����ûлܻ�������ܻû��������������ܻ������'�3�7�4�'�������ܻлԻܽ�����+�1�4�5�2�(��������������/�<�H�I�M�H�A�<�/�.�$�)�/�/�/�/�/�/�/�/����������������������������}�}����E�E�E�E�E�E�E�E�E�E�E�E�EuEiEbE^EdEiEuE��M�Y�_�f�r�������r�f�c�Y�O�M�@�<�:�@�M���������������y�r�l�d�`�]�V�`�l�y�������{�}ǅǈǋǐǈ�{�o�o�h�i�o�q�{�{�{�{�{�{�N�Z�g�u�t�s�g�b�N�A�5�(����%�'�/�=�N�����������������������������~�t���������������������ݽн����Ľͽнݽ���������������������)�<�@�@�=�)�������D�D�D�D�D�D�D�D�D�D�D�D�D�D{DpDnDwD{D�D����
�0�?�E�F�A�0�������ĿĴĨĤĨĳ���弱���ʼּ׼ּܼͼʼ���������������������ĳĿ����������������Ŀĳĳįĳĳĳĳĳĳ���ɺ˺ֺںۺֺӺɺ������������������������������������������~�x�r�p�r�s�~�������/�<�H�H�R�U�[�U�H�A�<�6�/�#���#�#�/�/ > " { G X ? . 1 T B 3 L A % V H H ` Y 3 d <  w f  ~ ' S x - 2 [ N " ( O A R Q n \ & 3 0 W ( K 5 J )    '  �  O  �  J  �  '  Y  �  $  Y  �  U  �  	f  m    �  b  V  �  �  �    ^  u  Q  ?  �  �  �  �  $  �  u    �     �  �  O  �  �  I  �  |  �  �  7    ���t��o�#�
<�1=�w=D��>1&�<49X<ě�=#�
<�C�<���<�j=C�>\)<�9X=��=8Q�<��=t�=o=���=��=t�=y�#=� �=@�='�=T��=H�9=��`=�;d=���=� �=}�=��m=ě�=�{=�\)=�hs=���=�hs=�x�>�%>O�;>!��=��`=�G�>1'>o>%B�RB17B"��A���B��B#ЎB}�B�eB ԓBOBEHB�&B�pB!ܯB}�B�Bg~B�
BBB!{�B%�WB�B��B��B!9B1B�B��B��B+ץBEB��B��B6$B5�BnHB|�BwxB,s�B�9BK B�!BePB	VB��B��B��BvB7B�#B�CB�[B@B"G+A�nbB�B#=�BDPBмB!X BR�BB�B��BXNB"@�BE�B�QBA�B�
BBB!��B%�3B
�DB$�B7\B!A$B=hB�BK�B:�B+��BG@B@�B?�BA�B@ B�QB��B�mB,@�B��B��BήBa�B	4�B��B�.B��B@!B �B �B��A��>A�"@�zA�r�AҎ�@�y(=�'A��k@T]]@���A���Als�@ihqA�b�B(�A��CAZ,�AE�A>��@��@�r}A��A��BA9oA&��A�U�A�!�A�q�?þ�@��Ay�@��
@��oA2ءA�.�A�n�C��@���A1�BB�A�ŉA��A-��Aӗ�C��#A��Y@�d�A�
i@,C�@oA�T�A���A���@��A�|�AҁN@�p�=�4A���@M9�@�lA�C�Am�@k�(Aˇ�B�KA��IA]�AEiA>��@��O@��pA�|A�PcA:�A'=wA��AȀA��t?�޳@���Azۮ@�r$@�ZA3 A��A�p*C��@�u�A�B?�A��|A�~�A/+gAӁQC��.A��@��MA�@+��@z@A©�               *   3   �                        y               
      2   L         6            
   <   @      &      F   +      	   	   
      %  )   x   I                                 3   -                        Q                     '   +      -                  1               )                        -      /                                                            !                     !   !                        +               %                              +               N/�N��N>5$O_^�N�/�O��2O�vGNO>�O=+N��NLGN��NE8�N��sO��{NF'�O]�JOT�N"�0NT:�N���O�I�O�FN��sO�1O�F�M�mN)ȯN��5N�Q�PO�O�'LN��O�H5N6�P
ٶOE�BN�V N�N�H�O{&yN���O]/�O�D	OB��P~4N��/N�]N���N�کN僱  �       Y  �  �  W    0  9  �    R    	  o  �  �  `  �  3  �  Y  �  "  	$  @  �  5  �  �  	w  �    E  �  �  A  @  �  A  �  /  M  �  
�  G  �    �  ༬1�e`B�T���o<���<ě�=���;�`B<t�<�j<T��<�t�<�t�<�t�=�^5<��
<�9X<���<���<�/<�/=#�
=H�9<��=49X=t�=\)=��=#�
=,1=H�9=T��=ix�=]/=P�`=��=]/=��=}�=�%=�+=�+=��w>C��=��`=��=�E�=�j=�/=�"�=�G�8ABNOQONEB8888888888������

����������������������������"'/;ADEB@;2/",./<HJUNH<1/,,,,,,,,������
'+*#
������a_^`enz���������znfambgnz�{znmmmmmmmmmm������������������������)-+))���������������������

�������
#)�������������������������!.342)����dbehotz}{tthdddddddd�����#$#����������������������)1+)��������������������  #.0<IQIG<60#      dipt�������������rjd��������
 #
�����spst�������������}ts��������������������yux��������������y������������������������

�����������T[_hmt����|tlhc[TTTT��������������������gmt�����������thlhig���������������2269BKO[[`a[ZPOB7622�#/<HQUYYUH</#
�~�����������~~~~~~~~������)266-)��������

 !
��������������������������������������������������������������������������������������������������������z}���������������~{z=9:>BN[gt�����tg[NC=���������
	������������'+*%�����nqsyz����������znnnn #$/;<?<://#����������������������������	���������������

 ������������������ŹŸŹź���������������������(�5�A�N�O�Z�O�N�A�5�+�(�%�'�(�(�(�(�(�(�����������������������������������������T�a�i�m�w�z�����z�m�a�T�;�,�"� �.�;�G�T��������������������������������������������������������x�l�_�T�S�Y�_�l�x�������ùϹܹ���޹йù����������������������������������������������������������������������ԺѺ޺����������������¼ȼƼ��������������������������<�H�O�P�P�H�<�4�1�0�<�<�<�<�<�<�<�<�<�<�y�����������y�m�`�Y�`�d�m�x�y�y�y�y�y�y��!�-�2�.�-�!�������������ÓàìíùúùøìàÓÇÅÇÈÏÓÓÓÓ����������	���������ƳƚƓƎƏƜƳ���������	�����	��������������������������	�"�.�4�.�,�"��	������׾;׾����s���������������������s�f�Z�V�W�Z�g�s�Z�`�f�a�Z�M�K�K�M�T�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z���'�.�0�'����
�����������'�4�8�@�L�I�H�@�6�4�,�'�$�#�'�'�'�'�'�'���	���!�"����	����������������������)�5�B�^�d�f�d�^�N�B�)�������������A�M�Z�\�_�W�M�A�4�(������(�.�4�?�A�����Ľн۽ݽ���ݽнĽ���������������ù��������������������ùìãÞØÚàìù�zÇÈÇ�|�z�n�n�h�j�n�t�z�z�z�z�z�z�z�z�/�;�H�J�L�H�;�/�+�,�/�/�/�/�/�/�/�/�/�/�@�D�L�U�Y�^�Y�Y�L�H�@�6�3�2�3�9�@�@�@�@�F�S�T�_�l�x�}�}�x�l�_�S�M�F�:�/�:�<�F�F�ݿ������	���޿��������|�}���������ݻ��ûлܻ�������ܻû��������������������	����������ܻۻֻܻ߻�����&�(�-�0�2�/�(�"��������������/�<�@�H�L�H�@�<�/�/�%�-�/�/�/�/�/�/�/�/��������������	� ����������������������E�E�E�E�E�E�E�E�E�E�E�E�EuEiEbE^EdEiEuE��M�Y�[�f�r�r�|�r�f�\�Y�W�M�@�>�<�@�F�M�M���������������y�r�l�d�`�]�V�`�l�y�������{�}ǅǈǋǐǈ�{�o�o�h�i�o�q�{�{�{�{�{�{�N�Z�g�u�t�s�g�b�N�A�5�(����%�'�/�=�N�����������������������������~�t���������������������ݽн����Ľͽнݽ��������#�-�0�1�-�������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D{DtDrD{D|D�D��������
�0�;�A�B�=�0��
������ĻİĬĳ�̼����ʼּ׼ּܼͼʼ���������������������ĳĿ����������������Ŀĳĳįĳĳĳĳĳĳ���ɺʺֺٺںֺҺɺ������������������������������������������~�x�r�p�r�s�~�������/�<�H�H�R�U�[�U�H�A�<�6�/�#���#�#�/�/ > " { G 2 :  1 T * 3 L A % . H H P Y 3 d -  w 0  ~ ' S m ' . 4 H ) ( O 6 R Q n \ &  + U ( K + J )    '  �  O  �  �  3  3  Y  �  #  Y  �  U  �  8  m    i  b  V  �  �  �    L  G  Q  ?  �  �  k  �  �  '  M  u  �  �  �  �  O  �  �  |  �  �  �  �      �  G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   G   �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  I  ,    �  �                
     �  �  �  �  �  �  �  k  U  >  '        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  Y  G  =  @  E  K  K  B  +    �  �  �  G  �  �    �  �  ;  �  @  �  �    H  s  �  �  �  �  �  �  ]  �  h  �  �      �  	  >  `  |  �  �  �  �  �  �  �  �  e  '  �  �  0  �  4  g  t  S    �  �    I  W  J    �  3  �  �    �  	�  �  U      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  0  "      �  �  �  �  �  �  �  x  k  X  =    �  �  n  �  �        !  ,  4  9  7  ,    �  �  �  n  0  �  �  "  L  �  �  �  �  �  �  �  |  v  p  d  R  @  .    �  �  �  �  c    �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  i  0  �  �  R  F  :  .  !      �  �  �  �  �  �  �  o  Y    �  Z       �  �  �  q  G  "    �  �  �  �  q  C    �  �  �  \  �  �  �  	  	  �  �  �  �  �  `  �  	  	  �  7  b  S  �  �  �  o  o  n  m  m  l  k  j  j  i  h  g  e  d  c  a  `  _  ]  \  �  �  �  �  l  M  0    �  �  �  �  j  j  E    �  �  N  O  �  �  �  �  �  �  �  �  �  �  v  Q  %  �  �  ;  �  Z  �  =  `  P  @  1     	  �  �  �  �  �  t  X  ;       �   �   �   �  �  �  �  �  �  �  �  v  m  i  i  p  ~  �  �  �  �  �  �  �  3  1  /  -  -  4  :  A  C  >  9  3  '      �  �  �  �  �  1  X  r  �  �  �  c  7    �  �  �  k  *  �  x  	  c  �  �  ^  �  �  1  S  V  G  /    �  �  `    �  (  �    H  E  '  �  �  �  �  �  �  �  �  �  �  �  �  �  |  t  m  X  ?  '        �  �  �        !      �  �  �  k  9  
  �  �  �  	  	#  	  	  �  �  s  7  �  �  ^    �  �  5  �  "  g  �  �  @  >  9  .    �  �  �  �  w  X  8    �  �  �  �  e  >    �  �  �  �  �  �  �  �  �  w  d  R  ?  -    
   �   �   �   �  5  .  '      �  �  �  �  �  b  D  %    �  �  �  m  ?    �  �  �  �  �  �  �  �  �  ~  \  9    �  �  �  ^  +   �   �  �  �  �  �  �  �  �  ~  R    �  �  D  �  �  T    �  P  �  	a  	r  	v  	n  	X  	;  	  �  �  }  9  �  �  /  �  5  �  �  �  t  w  �  |  o  _  L  o  �  �  �  �  v  C  �  �  B  �  q    T  �  �        �  �  �  �  �  �  e  >    �  �    �  �   x  5  ?  C  >  .      �  �  �  z  U  0  
  �  �  �  r  L  #  �  �  �  �  �  �  �  v  /  �  s    �  :  �  u  �  d    x  �  �  �  �  �  �  k  Q  +  �  �  B  �  �  !  �  /  U  j  n  
  7  ?  ;  *    �  �  �  |  D    �  �  M    �  p    -  @  9  2  ,  '    �  �  �  �  �  g  L  4      �  �  �  �  �  �  �  �  �  u  ^  H  %    �  �  �  P    �  �  [     �  A  ;  6  *    	  �  �  �  �  �  �  �  f  D  !  �  �  �  [  �  �    t  j  b  Z  R  O  Q  T  W  M  ;  )    �  �  7   �  /      �  �  �  �  �  i  +  �  �  >  �  �  S    �  �  B  �    _  �  e  �  .  L  1  �  b  �  y       n  �  �  �  "    F  r  �  m  7  �  �    �  �  8  w  �  �    �  	�  #  �  
n  
�  
�  
�  
c  
D  
(  
  	�  	�  	m  	  �  $  �  �  )  e  p  �  G    �  �    ?    �  �  :  �  �  Z    �  �  _  4    �  �  �  �  ^  9    �  �  t    �  P  �  �  B  �  q  �  B  �  �  
  �  �  �  �  ^  -  �  �  �  C  �  �  a  �  �    }  �  �  �  �  �  p  Z  L  ;  %    �  �  �  c  0  �  �  |  6  �  �  �  �  �  ~  d  :    �  �  |  H    �  �  �  P  !    R