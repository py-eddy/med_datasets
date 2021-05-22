CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�j~��"�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N"0�   max       Q        �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �'�   max       =��      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>\(��   max       @FP��
=q     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?޸Q�    max       @v�z�G�     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @P�           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�q@          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       >s�F      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��c   max       B1;�      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B1~R      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?bf   max       C���      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?`��   max       C���      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          Q      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          E      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N"0�   max       P���      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�-w1��   max       ?�ڹ�Y�      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �\)   max       =��      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�=p��
   max       @FP��
=q     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(��    max       @v�z�G�     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P            t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D   max         D      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�@N���U   max       ?�ڹ�Y�     �  QX      T      �   
   -   
   $                           I   �      1   	   %   .   C   
   	   1             �      �      >                  
   -   =                     �         l         '   
OQGP�M�Og�Q  N���P2�N�b�O���O��O�$�OŁ{N#/O��OאOXtOӟ{PUY�P�B�N"0�P"�N�,�O�G�O�*�P���N��	Nl�NP�\O���N�T�N�~�O��O��Pd�JN��P:bO��N�R�N�Z�NR@�O�<�Nh��O���P7hNK
�O\�On>�OqO)�N�L�P1,�O �OT�P�N�aN�OOG0�N-�ٽ'��
��o��o:�o;D��;�o<o<o<t�<t�<t�<49X<e`B<e`B<�o<�C�<�C�<��
<�9X<ě�<���<�`B<�h<�=+=t�=t�=t�=�P=�P=�P=��=��=#�
='�='�='�=,1=<j=H�9=L��=T��=T��=e`B=ix�=m�h=q��=q��=}�=�7L=��
=�-=�9X=�v�=�=��dbacgjtz��������trid����BNY]cb^N�����okrv�������������}woOR`uz��$+$����yUO��������������������$#)/<HUajrumRH/#������������������������������	
����� #&0<HISUWXUI<0#  "/;BHLMJC>/"�������
!
����������� ��������������� )-=CB6)�������������������������������

����T`m����������}trxg[T������
���������
)BN[t�������[B/
�������������������������)5BLIDB5)���������������������  
#(/33'#)6?BOPNKB6)�����5<UV;3.46)�������������������������������������������JKQ[h�����������h[OJ������"%%#
���������������������������#)*5752)����������

�������
 #02<HE<<50#
�toz���������������{tu|������������uuuuuuVWadz����������zmfaV;9:;CHTY^abdca\THB;;TUbdns{�����{nbUTTTT006CO\hhomh^\UOC;600NKLU`ahhaUNNNNNNNNNN����)5AEB8)���><=BOT[^[WTOIB>>>>>>/*6BEP[t���~|th[OB9/������/59??=5)	�������������������aUH<:/,)(-/9<HTX[`baTUUUV]amz������zmaZT�����������������������
#'$#
�������������
 ,*#�����utz��������������zuu��������������������{z�������		������{
��������
+'(,/<ABCA</++++++++#+/<ED><8/#HEHHUXa^UHHHHHHHHHHH�n�zÇÓÞàäàÙÓÇ�z�n�a�U�R�S�U�a�n�=�U�jŇşŔ�s�n�b�U�0��
��������
�0�=����'�@�3�/�-�'��������������s����������;�<�/��������s�[�G�5�N�g�s����������������޿��������"�;�T�a�l���������������z�a�H�����"����'�4�7�@�D�@�4�'����������������ûȻջܻ��ܻջû��������������������������������������������}���z�y��������/�H�T�]�a�`�R�H�B�;�/�"��	������	���� ���'�/�0�(������������������������������������������������������������3�@�L�Y�b�Y�L�<�;�?�3�'��������3���������ʼμռּڼּʼ������������������f�r���������������������r�q�f�]�[�f�f�������������������q�c�]�Z�P�Z�f�p�}�àù����������������ìÇ�]�D�@�H�U�aÇà��O�tČĢĨĦęč�t�h�O�6�-�"��
����F1F=FJFSFOFJF=F1F(F.F1F1F1F1F1F1F1F1F1F1���������������������������~�p�g�r�������#�/�<�A�?�>�<�/�#���#�#�#�#�#�#�#�#�#���׾��"�,�.�4�9�"�����ʾ��������������ʼ׼������ؼּ�����������������ƎƧ������9�$����Ƴƚ�u�6��� ���\Ǝ��� �	��	� �����׾ʾɾ��ʾ׾���������(�/�)�(��������������������������������Z�A�<�A�L�J�M�K�Q�f��4�7�A�E�F�K�V�M�A�4�%���������(�2�4àëìîùúùöìâàÓÇÅÆÇÓÛàà�Z�`�f�f�f�f�f�g�f�\�Z�T�M�C�D�G�M�R�Z�ZD�D�D�D�D�D�D�D�D�D�D�D�D�D�DsDjDmDuD{D������������������ݼؼ����~�������!�.�2�2�!����ֺǺ��������~�~�/�<�?�H�L�J�H�<�<�/�+�#�#�*�/�/�/�/�/�/�tāČĦĸļĻĳĦĚč�t�j�g�[�A�E�O�c�t�#�0�<�?�I�P�Q�I�<�7�0�#���
����#�#���������������������������������	���!�!����	��������������	�	���)�5�)�'���	�
�������������������������������������������������׺���������������ݺ����������� �'�4�:�D�C�@�4���ݻջջػջܻ�¿����������
���������·¦¦¬¿�[�g�t�x�t�g�[�N�K�N�Z�[�[�[�[�[�[�������������������z�y�v�l�a�]�]�`�l�w���� �����(�4�A�Z�^�f�h�`�M�4�(���ŇŔŠŭŹ������������ŹŭŠŒŇŃ�x�{Ň�����������ĿƿĿ��������������~���������y���������}�y�u�m�`�_�[�`�i�m�t�y�y�y�yECE\E�E�E�E�E�E�E�E�E�E�EiE]EXERENEAE;EC���
��������
�������������������������������������������������������������'�4�P�^�c�^�D�'������ܻݻ�����'ƧƚƎƁ�|ƀƁƍƎƚƟƧƧƨƧƧƧƧƧƧ����*�.�2�*����������������������
������
��������������������Ź������������ŹŭŹŹŹŹŹŹŹŹŹŹŹ Q / # J @ D S 1 A # q N i T , A = = J 8 > ` , � Q Q H < W Y  C J U 3 A � R : @ / F  h ? S 2 / r B b ' i = D 3 6    s  �  �  	  �  �  �  �  P  q  K  V  Z  W  *    �  n  2  �  �  �    Z  �  v  �  �  �     �  c  <  �  Z  F  �  5  Z  r  m  �  P  �  �    �  o  �  <  I     h  �  �  �  =����=��w<�9X>+<49X=@�<T��=49X<���<��<�h<T��<���=�w<�=C�=�Q�>s�F<���=�\)=+=y�#=��=\=�w='�=�{=�C�=]/=H�9>M��=T��>5?}=P�`=��`=]/=ix�=T��=Y�=�o=q��=��=�l�=aG�=��T=��w=��=��
=�\)>G�=�1=��>G�=�
==�"�>�->1'B	�B��BE�B��B�sBg@B!��B"�KB&"~A��cB�B5rBe�B L�B#��B
�BG	BiB �B��BȲBh�B6�B�B \B�HB�DB#��B"R�B!�B�B%?�B��Bo%B �A� �B(�B1;�B�B �B�dB_�B�5B�nB,�[B6gA��=B�B��BV�BM�B�B5�B��B��BڸB��B	�B�rBApB�UB�qB9�B!�}B"�yB&?�A���B6�B[B��B @�B#�FBA[B<�BZB��B��B�B��BٜBB�B DSBO�B��B$@@B"@BFB>�B%A�B�CB�kB �A���B(�+B1~RB��B�B�ABx�B�B�"B,��BƑA�|�BaBA�B��Bv�B��B��B�kB@�B�[B�9A�[�A�5H?bfA�}�A��A���@�΃@��
@� 3A�n|A���AJb�?�@�O@�DAF��A�ӳA�x_C���A��A�r`AX��@�(SB �AU�lA��LAD9�A7�A�'�A>`C��EA��@E/lA��A�`�A��@ZL�A[PqA��KA��?@QN�@��A�cqA���A��A8�A��YAsw�AlSC�<A�^�A��3@�qBA��A��A�P�A� A�j�?`��A�y�A���A�0]@�Y@�A�@�4�A�8A���AKV<?�@��@@�ڲAG�#A�nAڊWC���A�h�A�q�AX�&@��BH�AV��A���AD�A6�EA�v�A?�C��	A9�@H�3A��;A�}]A�	[@\hA[A���A��7@S��@���A���A�W�AfA9�A���As�Aj�C� eA�ÂA�b�@�^B��A���A�6�A�kV      T      �      -   
   %                           J   �      1   
   %   .   C   
   	   2   !         �      �      ?                     -   >                     �         m         (   
      7      Q      +               #               %   /   7      '      !      G         '                  5      %                     !   %                     -         +                  %      '                                       #                     E                           #                           !                        !         +            N��iP�%N��PS}N���O~޸N6��O/�O��O�U�N��%N#/O��N�/�Nf�iO�D�O�HO�w�N"0�O��'NyFDO�xOM?P���N��	Nl�NO|�O�nNF<N�~�OB��O��O�N{�NO��O��N:G�N�Z�NR@�O���Nh��O���O���NK
�N��cOn>�OqO)�N�L�O���O �OT�P�N�aN�OO$rmN-��    �  �  )  �  �  7  `  �    �  �  �  C  �  �  L  �  	  �  �  �  d    �  �  �  �    �  j  �  �  �  _  �  $  =  �  �  �  �  �  �       �  �  _  �  z  �  �  �  >  0  "�\)<���;��
=�+:�o<�1;�`B<�9X<o<#�
<���<t�<49X<�9X<�1<�C�=#�
=��<��
=\)<���=��=�P<��<�=+=aG�=�P=��=�P=���=�P=�{=#�
=u='�=49X='�=,1=@�=H�9=L��=�7L=T��=�7L=ix�=m�h=q��=q��=�"�=�7L=��
=�-=�9X=�v�=�=��fegjtz������tnhgffff�����)EKPNG5���tt}�������������xtt|����������������~z|��������������������-/3:<HU^aceka_UH?<2-������������������������������ ������ #&0<HISUWXUI<0#  	"/;AKLIB=/"	������

������������� ��������������� )-=CB6)��������������������������� 

���������[\ao�������������|g[��������������������)+05BN[gu|}{tgNB5,))�������������������������)39:1)���������������������
#'+/0/.$#
6BHJJFB96)����)5LT2-264)�������������������������������������������XUXcht����������th[X�������!#%%#
�������������������������#)*5752)����������

�������
 #02<HE<<50#
���������������������}�������������}}}}}}tpnoqv������������zt;9:;CHTY^abdca\THB;;bbhnv{�����{nbbbbbbb006CO\hhomh^\UOC;600NKLU`ahhaUNNNNNNNNNN�����)5?CB6)�><=BOT[^[WTOIB>>>>>>/*6BEP[t���~|th[OB9/�����/5774)�������������������������aUH<:/,)(-/9<HTX[`baTUUUV]amz������zmaZT�����������������������
#'$#
����������
!#!
������utz��������������zuu��������������������{z�������		������{
��������
+'(,/<ABCA</++++++++#'/9<@?<;6/#!HEHHUXa^UHHHHHHHHHHH�zÇÐÓÚÓÊÇ�z�n�a�Y�Y�a�n�q�z�z�z�z�0�<�I�U�b�p�t�s�l�U�I�0�������#�0����"�'�'�$������������������������������������������w�l�g�s����������������������޿��������a�z�����������������z�a�J�H�@�@�H�I�T�a���'�0�4�5�4�'�������������������ûлѻѻлϻǻû��������������������������������������������}���z�y��������"�/�H�T�\�_�^�P�H�;�/�"��	������	�������!�������������������������������������������������������������3�@�L�Y�b�Y�L�<�;�?�3�'��������3���������żʼʼʼ����������������������������������������}�r�m�r�t�����������������������������|�s�d�_�a�f�t�Óàìù������������þìÓ�z�]�V�[�e�zÓ�[�h�t�{ąĉĉă�t�h�[�O�@�7�2�1�5�B�O�[F1F=FJFSFOFJF=F1F(F.F1F1F1F1F1F1F1F1F1F1����������������������������x�{��������/�<�@�>�<�<�/�#���#�%�/�/�/�/�/�/�/�/����	���"�"���	������׾վϾ׾�𼱼��ʼּۼ޼�ۼּʼ�������������������ƎƧ������-�0���ƚ�u�6�*������\Ǝ��� �	��	� �����׾ʾɾ��ʾ׾���������(�/�)�(���������������s�������������������s�f�^�Z�Y�X�[�k�s��(�4�A�D�E�J�U�M�J�A�4�'���������àäìôìáàÓÇÈÓÞàààààààà�Z�`�f�f�f�f�f�g�f�\�Z�T�M�C�D�G�M�R�Z�ZD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DzD~D�D�D������������������ݼؼ���ﺽ�ֺ����!�!������ֺɺ������������/�<�<�H�J�H�H�<�5�/�-�%�%�.�/�/�/�/�/�/�h�tāčĚĦĲĴĲĩĦĚčā�{�s�o�f�^�h�#�0�<�?�I�P�Q�I�<�7�0�#���
����#�#���������������������������������	���!�!����	��������������	�	���)�5�)�'���	�
������������������������������������������������������������������ݺ����������� �'�4�:�D�C�@�4���ݻջջػջܻ��������
����
����������¹±³���������[�g�t�x�t�g�[�N�K�N�Z�[�[�[�[�[�[�y�����������������y�p�l�k�l�m�w�y�y�y�y�� �����(�4�A�Z�^�f�h�`�M�4�(���ŇŔŠŭŹ������������ŹŭŠŒŇŃ�x�{Ň�����������ĿƿĿ��������������~���������y���������}�y�u�m�`�_�[�`�i�m�t�y�y�y�yE�E�E�E�E�E�E�E�E�E�EuEiEdEaEbEfEmEuE�E����
��������
�������������������������������������������������������������'�4�P�^�c�^�D�'������ܻݻ�����'ƧƚƎƁ�|ƀƁƍƎƚƟƧƧƨƧƧƧƧƧƧ����*�.�2�*��������������������
�����
�����������������������Ź������������ŹŭŹŹŹŹŹŹŹŹŹŹŹ U ,  ' @ % N  A % B N i C " ? ,  J 2 = B - � Q Q 3 8 H Y  C / B ) A q R : @ / F  h : S 2 / r ) b ' i = D 3 6    �  �    �  �  �  c  q  P  L    V  Z  �  g  �      2  �  {  R  �  �  �  v  �  y  X     �  c    �    F  �  5  Z  5  m  �  u  �  �    �  o  �  �  I     h  �  �  g  =  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  �  �    
      	    �  �  �  �  �  3  �  /  �  	  v   �  %  �  �  #  P  v  �  �  �  �  |  z  m  D  
  �  >  y  l  �  �  
  1  N  d  w  �  �  z  j  U  ;    �  �  �  <  �  R  �  �    8  ,  #  j  �      $  &    �  o  �  ]  �  �  �  �  �  �  �  �  �  �  �  r  a  O  ;  '       �  �  �  �  �  �  �  �  �  �    @  f  �  �  �  m  M  &  �  �  ]  �  ^  �  �      '  +  .  2  5  7  7  ,      �  �  �  �  �  q  R  4  �  �  �  #  <  M  X  _  \  A    �  �  �  t  A  �  �  �  g  �  �  �  �  �  �  w  i  \  P  D  7  (      *  =  S  j  �  �    �  �  �  �  �  �  �  �  m  S  6    �  �  I    �  �  [  ]  b  �  �  �  �  �  �  �  �  �  �  �  �  j  :  �  �  b  �  �  �  �  �  v  q  k  f  `  ^  _  _  `  a  T  D  3  "    �  �  �  v  f  d  w  z  o  Y  9    �  �  �  f  (  �  @   �  �  �  �    +  <  C  >  5  )      �  �  �  >  �  f  �  ^  A  Z  o  �  �  �  �  �  �  �  �  �  _  9    �  �  �  �  �  �  �  �  �  �  �  {  ]  =    �  �  �  R    �  p  W  I  :  �  �  �    .  G  L  D  +    �  �  r  4  �  +  r  �  c  ;  �    '    �  f  �  J  �  �  �  N  �  	    �  -    	�  <  	                �  �  �  �  �  �  �  �  |  j  X  G  J  s  �  �  �  �  �  �  �  n  [  J  3    �  l  �  {    �  �  �  �  �  �  �  �  �  m  T  5    �  �  �  �  �  �  ;  �  �  $  [  �  �  �  �  �  �  �  m  1  �  �  ^    �  t  5  �  3  I  [  c  d  b  ]  O  5    �  |    �    �  �  �  �  �        �  �  �  �  m  �  z  K    �  �  }  )    n  �  �  �  �  �  �  �  �  �  �  �  {  j  X  F  1    �  �  �  T  )  �  �  �  �  �  �  �  �  m  P  1    �  �  �  �  e  C  !     o  �  �  �  �  �  �  �  �  �  �  �  f    �  [  �  6  _    �  �  �  �  �  V  -  �  �  �  S    �  �  z  0  �  <  �   �  �  �      �  �  �  q  B    �  �  s  3  �  �    �  #  �  �  �  �  �  }  u  j  \  L  4    �  �  �  �  \  2  
  �  �  �  �  \  �  �  5  ^  j  \  <    �     M  ?  �  a  �  
�  n  �  �  r  b  R  ?  *    �  �  �  �  �  �  z  Q  %  �  �  �  	�  
�  z  �  Q  �  �  �  �  q  J    �  5  
�  	�  �  �  �    �  �  �  �  �  �    f  M  5      �  �  �  �  ~  c  B    u  �  �    1  N  _  W  H  @  *    �  l  �  [  �  �  �  /  �  �  �  �  �  {  m  X  =    �  �  �  �  X  /      �  �  �  �      #      
     �  �  W  �  �  )  �  f     �  (  =  ;  8  .  #      �  �  �  �  �  �  �  e  C    �  �  k  �  �  �  l  O  .    �  �  �  �  �  G    �  �  �  ]  /    �  �  �  �  �  �  �  �  �  w  ]  @     �  �  �  �  a  G  �  �  �  �  �  �  �  �  �  �  �  y  `  K  9  8  @  Z  z  �  �  �  �  �  �  �  z  Z  4    �  �  ]    �  /  �  �    y  �  c  �  �  �  �  �  �  �  �  �  ^  2    �  h  �  i  �  �  -  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  V  ~  �  �  �  �            �  �  �  �  C  �  �  �      �  �  �  �  �  �  }  T  #  �  �  f    �  �  �  ;  �   �  �  �  �  �  �  �  �  �  }  `  7  �  �  T  �  �  <  	  �  �  �  �  �  {  l  \  H  2    �  �  �  X    �  �  l  '  �  �  _  E  )    �  �  �  b  /      �  �  J    �  k    �  �  T  �  P      �  �  �  �  �  S  �  Z  �  �  �  
:  �  �  �  z  x  s  j  ^  O  >  )    �  �  �  h  ;  
  �  �  A  �    �  �  �  �  p  _  N  =  ,      �  �  �  �  �  k  R    ~  �    q  Z  4     �  l    �  �  j  �  	  
1  	H  M    �  l  �  �  �  �  �  c  3    �  �  h  5    �  �  f  A  &      >  2  &      �  �  �  �  �  o  S  7      �  (  =  A  C      *  /  &    �  �  �  �  u  =  �  �  K  �  X  �    \  "    �  �  �  �  �  i  H  &    �  �  �  5  �  �  &  �  [