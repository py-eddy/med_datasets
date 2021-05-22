CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�x���F      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       P��Y      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       >o      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�����   max       @FXQ��     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���R    max       @vmp��
>     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q@           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �49X   max       >w��      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B2 �      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��x   max       B1�(      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?I�	   max       C�
�      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?O�   max       C�      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       P�=�      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�N;�5�Y   max       ?�P��{��      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��j   max       >o      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�   max       @FXQ��     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?޸Q�    max       @vh          �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @O�           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��@          �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Mj   max       ?�P��{��     �  QX      ^         +                  .            �   T      	   3   L   6   .      g                  �   >                     �      *                  (   2      �                     �   N�SP���ND�]N���O��XN���O��N�O�l�N��O�MwN�o�N*��O�ƘPCn�P<��OcO!�PW�Ps��O�eTO�N"��P��YN_1�N��)Nt��N�*�N�m&PQ�P��No�gO��O��zN ��N��NN�"SP��4OcdpO��{OGD�O:�.O\��N]�RN�O�e�P�N؟�P0�N���N�W�NjhN�z=N���N�jO�T�O4���ě��ě���o%   ;D��;�o;�`B<t�<t�<D��<e`B<e`B<u<�o<�o<�o<�o<�o<���<���<���<��
<��
<ě�<ě�<���<���<�/<�`B<�h<�=o=+=+=C�=\)=�P=0 �=8Q�=8Q�=<j=<j=<j=@�=P�`=]/=aG�=aG�=u=�o=��=�+=�+=�7L=�O�=��P>oqnmrt~����������|tqq�����'$/BTZN5���rmpt�����trrrrrrrrrr'%&)5BNRNIDB5)''''''����������

������������������������������������������������������������mlt��������������xtm�������������������������45BB6)���������

���������rtt�����|trrrrrrrrrr!"&/9HUZcghfa]UH</$!��������  ��������:;7<B[����������t[B:�������������������������������������������������������%$,7[h���������thO6%����
060+#!
�����������)5DB5)����_`gt���wtg__________Vbnz��������������aV����������!"(/;HKHG?;/$"!!!!!!:9<HU^aeaUH<::::::::	�
##,0100)#
		������

	

������!)B[t�������t[5������)5HVZJ���������
������������)45750)����sx���������������~zs?>BO[\\[OB??????????UQPU\ehjuuuuph`\UUUUdno|�������������{ndx|{������ ��������x81/1;HTU\adfeejaH?;8ihjmz�����������zpmiiffdgmz����������zmi	
#/5;><6/#
*(+,/<HMU[cfcaUH<7/*8;@BO[^[ZQOB88888888��������������������������������������������)5>@?5	��������������������**$��������������������������������~����������)-.-+)��������������������HHE<<;/.+*/2<FHPHHHH��������

�����		
#$.00.*#
	�n�zÇÓØàäàÛÓÇ�z�n�j�a�_�a�h�n�n�#�<�UŔŲ��Ź��ŹŭŇ�n�b����������#�@�L�Y�]�c�Y�L�@�4�=�@�@�@�@�@�@�@�@�@�@�G�T�`�c�l�g�h�`�[�T�G�F�@�E�G�G�G�G�G�G���������ûܻ������ܻл�����������������������
��
�����������������������������!� �������ܹϹ̹Ϲӹܹ�����B�N�[�^�[�[�N�B�5�0�5�7�B�B�B�B�B�B�B�B��"�3�=�@�;�.�"�	����׾Ӿξ;׾������������������������������� �������ּɼ����������üʼּ㼋������������������{�v������������������û̻λû������������������������������a�m�z�������������z�m�a�Y�T�H�@�B�H�T�a�ɻ�(�*�)�&�����ﺽ�����������������ɻ���'�Y�f�l�g�a�`�M�@�+����ܻѻ׻ۻ����������ǼʼӼټּҼʼ������������������4�;�@�B�H�M�P�I�@�9�4�*�'�����'�1�4àù��������������������ìÇ�`�Q�J�`Çà�f�s���������������f�Z�A�8�.�*�,�.�;�L�f�A�D�J�Z�g�j�k�f�A�(�����������)�A�������������������������{�v�y������������������������������������������������������������������������g�Z�N�;�E�Z�b�v�����	�����	���������������������������a�c�m�s�w�u�m�b�a�T�J�O�T�W�a�a�a�a�a�a������� ��������������������������������ּ����������������ּ̼ʼԼּ��#�/�<�>�H�T�H�D�<�2�/�-�#����#�#�#�#���O�hāċĢĥęā�h�F�6�.�2�1�#���������������.�.�'����������������������3�@�L�V�Y�\�Y�L�@�3�/�0�3�3�3�3�3�3�3�3�Z�f�g�j�n�m�k�f�d�Z�V�M�D�F�G�G�M�P�Z�Z�Ŀѿڿ����ݿĿ��������������������Ŀݿ�������ݿݿ׿ۿݿݿݿݿݿݿݿݿݿݾʾ׾��������������׾ξʾȾʾʾʾʺ������!�%�+�-�/�-�!��������������Y�f������r�Y�E�'���лû̻ǻлܼ�'�Y����#�0�7�5�1�0�#���
����������������ĚĦ������������ĳĦč�~�t�r�v�s�wāĊĚŔŠŭ����������������ŹŭŨŠŖŔŌŎŔ�`�m�y����������y�m�`�S�G�<�;�G�I�T�]�`�A�M�f�s�}�t�j�f�^�Z�M�A�4�2�&�!�#�(�2�A�������ĺĺ�����������������������������ÇÓàéèåàÓÇ�|�}ÃÇÇÇÇÇÇÇÇ�����������������������y�l�`�W�U�e�l�y��¿���������������������´¦«µ¿�[�g�h�p�t�z�t�g�[�X�N�L�K�N�N�Z�[E\EiEuE�E�E�E�E�E�E�E�E�E�E�EuEiEbE\EPE\������������������������������������}�����
������������
���� � �#�)�#�#�ƎƚƠƧƱƧƚƎƁ�u�o�uƁƂƎƎƎƎƎƎ��������� ������������������������������Z�_�g�s�|�~�t�s�g�Z�W�S�R�U�Z�Z�Z�Z�Z�Z¦¥¦²½¿��¿¼²¦¦¦¦D�D�D�D�D�D�D�D�D�D�D�D�D�D�D}DzDD�D�D��������#�0�@�<�0�#�
������������������ > @ : : 6 ` E n X p S ) )  2 7 > N - = 8 T m 9 & < V  A C C & Z ' > + j R Q 9 / 5 9 N < 7  G , � w u : 9 ; + m      �  X  �  �  	  e  D  �  `  �  �  =  
  W  +  i  X  �    +  ?  ~    h  �  �    �  W    y  �  s  J  �  }  y  �  �  �  �  �  �  �  �  8    �     W  �  �  �  �  m  ϼ49X=�9X;D��;ě�=<j<u<���<D��<��<�o=u<�1<���=�P>A�7=���=#�
<���=�\)=��=��=��<�9X>J<�h<��=t�=�w=\)>ix�=�j=��=0 �=@�=��=�w=H�9>]/=�C�=� �=��=�hs=�O�=T��=�O�=�v�=���=}�>?|�=�hs=��
=�O�=�\)=�hs=� �>w��> ĜB
*�B��B�RBf�B# B��B��BB^�By5B�B#��B��B<BQ:B�IB PmB"gB;�B5�B$M�B�$B	��B^lB�A��B^_B$��B0zB~B��B��B�BxrB��B2 �B)<�BۭA��B ��A��B�B
B��B!��B,^�B��Bf+Bg�BIoBn�BހBd
B	�B�{B�`B��B
@�B��B��BC"B#?�B��B^�B�CB�B̺BYB#�VB�BD5B@B�%B @B"�B��B,*B$@�B�#B	��BXFB<CA��xB@B%<�B��B��B��BP�B��B�B�BB1�(B)8�B9�A�~B �B ?:B��BϻB~�B">�B,P�B��B|�BK*BSBR�B�B��BC�B�{B�jBY�A���A��?���Ag��@��A�L�?I�	A��A[�A� 	A/E@��@��A�u9@G�@��@���@�|�A˒AB�XA7��A�ʀAH+RA�xA�1fA���A��
At�A��5A���A�.�?�QA?32AwIJA~W*AT�x@^"$@�`�A�9A��A�=@Ai��A<�g@$SA�R�A��A���A�>C�
�A���A��B�$BL�A�Q�A��>C��A�*dA�i�A�}?�cAg ^@�ƐA��I?O�A�.jA\�A���A�D@���@���A�#Z@K��@�X�@�:�@ΘA���AED�A7�A�byAI�A�}�A�wA��A�qIA�A�{nAڔ�A��?�-aA?Av$�A~�AT�(@Z�@��A胔A��MA��iAj�@A<�4@#��A�{A�A�i�A���C�A�ۇA��B��BL�A��TA�s,C��A� �      _         ,                  /            �   U      
   4   L   7   .      h                  �   ?                     �      *                  (   2      �                     �         7         !            !                  )   -         /   1   %   '      =                  3   7         !            5      !                     #      )                                                                                 '   %            1                     5                     +                           #                              N���O�D�ND�]N���O*YN|צN�̠N�OFN��O��%NHc�N*��O�ƘO���Os�JNI�,O!�P$�1P��O��Om��N"��Pw�4N_1�Nu;�N)ON�*�N�m&O���P�=�No�gNs�<O���N ��N��NN�*aP.�#OcdpOs�OGD�O:�.O\��N]�RN�Ol�Oܫ=N؟�OQ:�NB��N�(RNjhN�z=N���N�jO�i�O4��  E  �  q    �  I  �  �  �  �  V  3  B  <  c  �  �    �  `  �  �  R  �    �  "  a  8  ]    4  �    I  �  f  �  v  �  )  x  �  �  ^  �  Q  �  �  T  �  i  �  �  �    	}��j=<j��o%   <�1;ě�<49X<t�<�t�<D��<�C�<u<u<�o=�^5=ix�<���<�o<�/='�=+=+<��
=49X<ě�<���<�/<�/<�`B=�h=C�=o=�P=C�=C�=\)=��=���=8Q�=aG�=<j=<j=<j=@�=P�`=�O�=y�#=aG�=�h=��=�7L=�+=�+=�7L=�O�=��T>oqonst�����������}tqq�����)+38:75)�rmpt�����trrrrrrrrrr'%&)5BNRNIDB5)''''''����������������������������������������������������������������������������tv���������������{tt������������������������)2>?6)���������

�����������rtt�����|trrrrrrrrrr!"&/9HUZcghfa]UH</$!��������������������NKIIIO[htz����|th[ON�������������������������������������������������� ��������A:88?I[h���������h[A������
#'++*#
����� )15751)����_`gt���wtg__________gafnx�������������yg����������"")/;HIHE<;/)""""""";:<HU[aUH<;;;;;;;;;;	�
##,0100)#
		������

	

������.+)+0BN[gt|��}tg[B5.������)5EQSD���������
���������')1)&|w{����������������|?>BO[\\[OB??????????UQPU\ehjuuuuph`\UUUUfnp{}�����������{nff��������������������81/1;HTU\adfeejaH?;8uqopuz������������{uiffdgmz����������zmi	
#/5;><6/#
*(+,/<HMU[cfcaUH<7/*8;@BO[^[ZQOB88888888����������������������������������������������,5;=<5)����������������

���������������������������������������������)-.-+)��������������������HHE<<;/.+*/2<FHPHHHH��������

�����		
#$.00.*#
	�n�zÇÓÕßÔÓÇ�{�z�y�n�m�a�`�a�i�n�n�#�0�<�I�V�c�i�d�X�U�I�<�0�#�"�����#�@�L�Y�]�c�Y�L�@�4�=�@�@�@�@�@�@�@�@�@�@�G�T�`�c�l�g�h�`�[�T�G�F�@�E�G�G�G�G�G�G�������ûλлջѻлû�������������������������� �
��
������������������������������������������������B�N�[�^�[�[�N�B�5�0�5�7�B�B�B�B�B�B�B�B��"�.�/�5�.�*�"���	������� �	������������������������������������ּʼ¼����������żʼּ���������������}�y�������������û̻λû������������������������������a�m�z�������������z�m�a�Y�T�H�@�B�H�T�a�ֺ�������������ֺɺ��������ɺּ���'�4�A�K�P�O�M�I�@�4�'������������������¼������������������������������4�;�@�B�H�M�P�I�@�9�4�*�'�����'�1�4àìù��������������ìÇ�k�[�X�_�m�zÇà�Z�f�s������������������s�Z�M�C�@�A�F�Z���(�4�A�M�Z�]�a�^�Z�M�A�4��������������������������������������������������������������������������������������������������������������������g�S�L�M�g�s�����	�����	���������������������������a�a�m�r�v�q�m�h�a�T�R�R�T�`�a�a�a�a�a�a����������������������������������������ּ����������������ּ̼ʼԼּ��#�/�<�>�H�T�H�D�<�2�/�-�#����#�#�#�#�B�O�[�h�tćčĎĉā�t�h�[�O�A�6�4�5�:�B������������+�+�!����������������������3�@�L�V�Y�\�Y�L�@�3�/�0�3�3�3�3�3�3�3�3�Z�Z�f�i�i�f�e�Z�N�M�L�M�M�Y�Z�Z�Z�Z�Z�Z���Ŀѿ׿ݿ���ݿܿĿ������������������ݿ�������ݿݿ׿ۿݿݿݿݿݿݿݿݿݿݾʾ׾��������������׾ξʾȾʾʾʾʺ�������!�$�(�!������������������M�Y�l�r�m�b�M�:�'����ֻܻԻѻӻ�����#�0�7�5�1�0�#���
����������������čĚĦĳĿ������ĹĳĦĚčā�~�y�|�{āčŔŠŭ����������������ŹŭŨŠŖŔŌŎŔ�`�m�y����������y�m�`�S�G�<�;�G�I�T�]�`�A�M�f�s�}�t�j�f�^�Z�M�A�4�2�&�!�#�(�2�A�������ĺĺ�����������������������������ÇÓàéèåàÓÇ�|�}ÃÇÇÇÇÇÇÇÇ���������������������y�l�h�d�`�_�`�l�y������������������������¼­¤¦°¹¿���[�g�h�p�t�z�t�g�[�X�N�L�K�N�N�Z�[E�E�E�E�E�E�E�E�E�E�E�E�EuErElEmEqEuE�E������������������������������������������
���������
�����������
�
�
�
ƎƚƠƧƱƧƚƎƁ�u�o�uƁƂƎƎƎƎƎƎ��������� ������������������������������Z�_�g�s�|�~�t�s�g�Z�W�S�R�U�Z�Z�Z�Z�Z�Z¦¥¦²½¿��¿¼²¦¦¦¦D�D�D�D�D�D�D�D�D�D�D�D�D�D�DD|D�D�D�D��������#�0�@�<�0�#�
������������������ C ! : : ! L   n I p U 0 )   , ) N & 2 . 3 m & & ; Z  A  > & R * > + g E Q + / 5 9 N < .  G  ~ W u : 9 ; * m    �    X  �  F  �  �  D  d  `  g  V  =  
  '  �  [  X  �  �  E  �  ~  �  h  �  d    �  �  �  y  �  B  J  �  ,  P  �  �  �  �  �  �  �  a  �    �  �  �  �  �  �  �  Y  �  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�    8  :  &    �  �  �  �  �  h  L  3        �  �  �  a  �    }  �    5  T  a  j  y  �  �  x  \  '  �    e      q  �  �  �  �  �  �  �  �  �    {  v  r  o  o  n  h  _  W      �  �  �  �  �  �  �  �  �  �  �  �  y  a  H  ,     �  �  (  h  �  �  �  �  �  �  �  �  z  G    �  b  �  |    /  <  8  9  D  @  .    �  �  �  �  �  �  �  �  �  �  �  �  �  3  7  5  9  �  �  }  m  Z  B  #    �  �  �  o  .  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      0  �  �  �  �  �  �  �  �  �  �  �  �  ^  2    �  �  5  �  �  �  �  �  �  �  �  �  x  n  d  _  ^  ^  ]  ]  S  F  9  ,    J  V  J  4    �  �  �  �  }  O    �  ~    �  �  L  �  �      )  5  >  G  M  Q  U  M  D  8       �  �  �  y  T  /  B  G  K  D  <  4  ,  $                    �  �  �  <  .  %           �  �  �  �  �  o  F    �  �  J    �  
Z  )  H  7  �  �    I  a  R  '  �  �  �  �  �  q  	�  f  6  �    -  N  z  �  �  �  �  �  �  �  �  �  U  �  O  �  �  �  B  a  |  �  �  �  �  �  �  �  �  �  �  u  <  �  �  4  �  W                  �  �  �  �  �  �  �  �  �  �  h  G  �  �  �  �  �  �  �  �  �  �  U    �  m  =  $  �  �  �  �  �  �    8  N  \  ^  W  U  K  9  "    �  �  &  �  �  �  T  j  �  �  �  �  �  �  �  �  �  �  a    �  H  �    \  p   �    �  �  �  �  �  �  �  �  �  �  b  2  �  �  I  �  A  �  �  R  M  H  C  =  8  3  .  )  #               �   �   �   �  �  <  i    �  m  H  2  "      �  �  �  :  Q  {  )  O  �         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  w  l  I  &            8  b  k  m  f  ]  N  A  5  '      �  �  �  �  a  Y  O  B  /      �  �  �  �  �  ~  e  F  )      R  �  8  )      �  �  �  �  �  �  �  �  �  �  �  �  �  �    J  �  g  �  _    ~  �  '  X  L    �  D  �  �  w  $  b  �  �        �  �  �  r  U  )  �  �  !  �  u    �  y    c  a  4  '        �  �  �  �  �  �  �  p  T  5    �  �  �  J  �  �  �  �  �  �  �  �  �  �  �  �  �  `  $  �  �  ]  6            �  �  �  �  �  �  �  �  �  w  f  S  <    �  p  I  A  8  0  (          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  q  g  ]  R  E  9  -  !  M  \  \  G  1        �  �  �  �  �  �  k  J     �  d  �  /  �  n  �  �  �  �  J  �  t  �  w  �  �  �  
v  	!  �  �  �  v  o  g  ^  d  T  >  $    �  �  �  g  -  �  j  �  G  �   �  �  �  �  �  �  �  �  �  �  �  w  A  �  �  P  �  {    �  }  )  "      	  �  �  �  �  �  n  <    �  �  �  j  C    n  x  k  X  A  &    �  �  �  W  "  �  �  X    �  ?  �  �  4  �  o  f  g  j  k  i  f  `  X  I  7  %    �  �  �  Q  �  �  �  �  �  �  �  �  �  �  t  ]  F  0      �  �  �  �  �  u  ^  S  ?  &    �  �  �  �  �  �  �  w  U  /    �  �  X    R  h  �  �  �  �  �  �  �  �  �  �  �  �  �  H  �  �  8  �  F  M  Q  L  :  $    �  �  {  <  �  �  +  �  ?  �  ;  �  ]  �  �  �  �  t  e  R  <  &    �  �  �  �  y  \  ?    �  �  
�  �    �  �      �  �  �  �  C  �    /  
  
�  	  <  �    #  @  S  N  H  D  A  =  @  C  F  H  I  J  K  L  I  D  ?  �  h  �  �  �  �  �  j  :    �  �  x  B  	  �  �  T  ?  �  i  `  W  N  D  ;  2  *  #                  "  )  0  �  �  �  z  s  k  _  T  H  =  1  &          =  ^    �  �  �  �  �  �  �  �    x  r  l  e  ^  W  Q  E  8  +      �  Y  2    �  �  v  B  �  �    8  �  �  !  �    v  �  -       �  �  �  �  X    �  Q  �  X  �  �  ,  �  �  �  z  H  	}  		  �  �  �  ]  ,  �  �  �  i  8  �  �  h    �  ]    