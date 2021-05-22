CDF       
      obs    A   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�|�hr�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�@�   max       P��k       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��w   max       >�u       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @F�\)     
(   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vXQ��     
(  *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @P�           �  5   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�i@           5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��o   max       >S��       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�3�   max       B/,�       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B/@�       8�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?E\�   max       C��e       9�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?PY�   max       C��5       :�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       ;�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9       <�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +       =�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�@�   max       P�K       >�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�`�d��8   max       ?�=�b��       ?�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       >�u       @�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @F�\)     
(  A�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vXQ��     
(  K�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P@           �  V   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @� @           V�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @R   max         @R       W�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?wX�e+�   max       ?�=�b��     P  X�   �                              	   -         K      +         :   '   	                  �         	      *      $            $   /                           
                     
            
      0   0      .P��kN���N�,�OK��NC�NZ�O12WN
�UNEEN;�N��O��Nu/�NA�Pk��N��O���M�@�N��$P�ŅOZ0|N�ӈNɦ>N�F�O2�O؆�O(�GO�ߵO���O0�JO$�N_Q�Ox	N�nO�%zN'��N�R-N-O@ �P 6OzeN �N�]|O�|�N
TN�5LN�-O�NWN�]OB�M��O�N�#N��N$k4N�(�N8�O9'�NAw�Nw�O#�O:O���N%0ZO>��w��/�ě���9X��1���
��o�#�
���
�D��%   %   :�o;o;D��;�o;�`B;�`B<t�<#�
<49X<�t�<���<���<��
<�j<�j<���<���<���<��<��=o=o=C�=C�=t�=�P=��=��=��=��=�w=0 �=0 �=8Q�=8Q�=<j=<j=D��=Y�=]/=e`B=��=�7L=�O�=��=���=��
=��=��=�E�=Ƨ�=�
=>�u#BNg��������tc5&%-)*-0<EFHG<0--------���������������>?>==@BN[fgklkkg[NB>����������������������������������������}������������������}������������������������������������������������������������������������������������������������������������������������FHO[hihh[OFFFFFFFFFF#<PURUa�������nUH/{����������!)5BNt{�{ncNB5)!~���������~~~~~~~~~~#"##,//028<<<:4/)###AR[ew�����������g[HA����
#+)&#
����QLJSTamyxtmlaTQQQQQQ���		���#0<?A<<20'$#)36BSQOOXOB6)�������������������@@BLO[bfiklkh[YOLGC@������
#%''$
�����������������������������(($���.-/)'/<?CHPVSSHD<4/.��������������������#/<HUaca\UH<4/#����������n������������������n����������������������������������������������������������������������������
#/<EGGD</#���25:BFOZ[_bca_[OJB962Y[hjt���yth[YYYYYYYY)*5BDEB>65)��$)*%)'"!!���&'))57765)&&&&&&&&&&�)-0/)%����� 
")+--)������).158CDB5-�55:BN[`c[UNB65555555����%))$������{x{��������������������������������������*/6776/*����������������������������������������������
##$&#
���)(),36A9766)))))))))��
#).2<AEC</(#
_\Z]akmnnona________::;=HTZacaYTIH@;::::``a^\amz~����}zomca`��������������������wz����������������{w��������������������jhnz���������|zsonjj���6�P�R�G�)�����ùìÇ�v�uÇàù����Ľнݽ�����ݽнĽ����ĽĽĽĽĽĽĽ�Ź������������ŹŭŪţŢŭůŹŹŹŹŹŹ���������
���� ���
�����������������{ŇŔřŚŔŇ�{�r�y�{�{�{�{�{�{�{�{�{�{�'�(�3�?�<�:�3�,�'�!��$�'�'�'�'�'�'�'�'�������
����
��������������������������������������������������������t¦§¦�t�m�t�t�t�t�t�t��������������t�z����������������������;�G�T�`�l�m�m�m�c�`�T�G�;�6�3�:�;�;�;�;����A�J�O�M�A�4���������������¦²¿��������¿²¦£�ּ�����ּԼ̼ͼּּּּּּּּּֿ��	���5�A�Y�Y�}�Z�N�(������׿ڿ��*�6�>�7�6�*������(�*�*�*�*�*�*�*�*�;�G�`�������������y�m�`�T�G�;�-�'�(�+�;�Y�e�e�r�e�e�e�Y�X�V�Y�Y�Y�Y�Y�Y�Y�Y�Y�YD�D�EEEEE EEED�D�D�D�D�D�D�D�D�D��5�g�t�������������s�N�A�(���	�����5�����%�)�/�2�1�)��������������������H�T�a�m�n�u�p�m�a�T�I�H�<�B�H�H�H�H�H�H�/�6�;�H�P�T�`�T�H�;�/�"�����"�-�/�/�������������r�r�f�`�f�r�s������àêççìï÷ìàÜÐÇÄ�|�w�{ÀÇÓà��"�/�3�;�T�a�e�a�U�;��������������������ʾ׾ؾվʾ������������}�����������D{D�D�D�D�D�D�D�D�D�D�D�D�D�DsDeDbDbDoD{�ѿݿ�����'�+�,�(������пſ����Ŀ��m�z�������������z�m�a�T�S�H�C�?�H�T�`�m����������������������������������������ààìõùÿùóìàÛÙÛààààààà��������������������������������뻞�����ûлӻܻ޻�ܻлû��������������������#�)�"�����ܻû����ûлܻ����ݽ������������ݽܽݽݽݽݽݽݽݽݾ��������������s�p�m�o�s�z�������"�/�0�;�D�H�I�H�B�;�/�"� ���!�"�"�"�"�a�n�zÃÇÊÇ�z�r�n�a�U�F�<�/�<�>�H�U�a�������׾���%�&����׾����������������ܻ�������������ܻڻһлͻлԻܼ'�)�1�4�9�8�4�0�'�!���'�'�'�'�'�'�'�'��#�(�2�4�>�7�4�(�����������Ƴ���������
����������ƧƚƎ��zƂƎƚƳ��$�0�1�8�0�$����������������������
�������������������������������������	������	������������������#�0�<�E�U�[�\�U�<�/�#���������������n�p�{ŀłŁ�{�n�c�b�`�_�b�e�n�n�n�n�n�n�_�a�l�w�l�_�S�F�:�-�!����'�6�:�F�S�_������������������������～�������ĽƽĽĽý����������������������T�`�h�m�y�}���������z�y�m�k�`�[�S�Q�Q�T�)�6�B�O�S�[�^�[�R�O�B�6�)�(� �#�)�)�)�)�"�$�.�5�4�.�"������"�"�"�"�"�"�"�"�����ʾ׾۾�׾Ҿʾ����������������������~�����������������~�}�|�~�~�~�~�~�~�~�~ǔǡǭǹǵǪǡǔǈ�y�o�b�V�U�`�o�r�}ǈǔ�����ûлܻлû��������������������������h�t�w�t�o�j�h�^�[�Y�O�J�O�O�[�e�h�h�h�hŹ��������������������ŹŬŠŝŠšŭŶŹ²¿������� ����������¿²®¦£¡¤¦²��������������������������r�h�^�`�f�r�E7ECEPESEXEPECE=E7E.E7E7E7E7E7E7E7E7E7E7�ɺֺ�ݺݺٺֺɺ��������������������ɺ� < 1 < : 4 \ " a v S * 2 K F h � + _ h 8 8 + Y ; M . L + 1 , _ X 0 _ L x : . 6 R < ~ G s g ; e W I e Y  S - { E t � i Z Q : f \ *    �  �    �  O  �  �  G  �  g     �  �  x    :  :  ,  �  ~  �  �    �  �  �  �  �  �  |  ]  �  �  '  }  u  �  �  �  �  j  �  �  �  /    B  �  �  �    <  ,    �    R  �  h  �      �  U  :>S�ϼu�e`B;��
��o�e`B�D�����
��o:�o<o=8Q�;�o<t�=��-;ě�=L��<#�
<�j=�7L=H�9<���<�/<�/='�=@�=,1>%�T=Y�=#�
=�w=��=��P=8Q�=�\)=49X=,1=49X=��=�1=u='�=8Q�=ix�=<j=�o=Y�=��=e`B=���=u=�+=�O�=��-=��P=��w=���=ě�=�^5=�j=��`>I�>z�=�h>G�Bq B&:Bh�B��B"B ��Bt�BfpB�HB)�?B;>B!
�B�B
B'bB
�wB#2B%OB\B
d�B��A�� A�3�B%�SB��B�<BB�B�qB��B�B!��B�PB�)B ��B)�vB��BB�
B]�B�B��B�qB�QB?B$�B�IB��B�B�B)E>B,��B/,�B��B,�tB$v�B��BT�B[A��'A��$B��BV�BJgB�PB@�B&�BGB�_B/B �BzTB�NB��B)��BJB ��B zB�;BHB
��B)�B?�BBRB
6jB�~A���A��B%�
B��B8�B<!B�AB�IB��B�B!�'B�*B��B ��B)A�B��B�xB��B�jBC�B8NB@B��B��B@�B��B4�B�:B��B)=@B,��B/@�B�eB,�!B$��B��B@dB<�A�u
A�{*B4�B��B>aB��A�!FA*-VA��rA�%�A��?��nA�p?E\�A�G�@���Af��A4�nA�~OA�0A��UA�:Ahظ?���C�GXA��,A���A��A�+�@���AʛA��AK��C�ΥA��A�.lA���A��>A�b�@�Ib@��eA.��AD�=A�rUA�l�AS@��@ˢ�A6fB��B	��A�A��A�'�A�@��@V�?A"Z�Aj\AA�2VA_�AN��@�sBS�@�#	A� `A��"A��6@�:C��e@0\A��A+ A��"A���A�f?�׆A��&?PY�A�}S@� fAee�A5�A��oA
�A�gA��Ai?�C�GZA�u]A���A��"A�n�@��.A�z�A��qAL� C��A���A�T�A�>A�U�A�}�@�&s@�)�A/AE A��7A��AS~�@���@ʄA7�B9B	^	A�{�A�}LA�G�A��"@��!@T rA#!Ak�Aע)A^�AN^@��BI�@�C8A�`A�pXA��n@��C��5@,y�   �                              	   .         K      ,         :   '   	   	   	            �         	      +      %            $   /                     	                           
                  1   1      /   9                                 !         9      !         9                  !                           #               '            #            !                                             #                                                                  +                  !                                          '            #            !                                             #      O���Nb�PN�WVONC�NZ�N�j�N
�UNEEN;�N��O_�UNu/�NA�O��N��O��EM�@�NHhQP�KN�
|N���Nɦ>N�F�N�\�O�T)OեOD�fO��O��N���N_Q�O ��N��COa@ZN'��N�R-N-O@ �O���N�\YN �N�]|O�|�N
TN�N�E3O�NWN�]O6~�M��O�N�#N��N$k4N�uN8�N�T�NAw�Nw�O#�O:O���N%0ZO>�    6  �    �  4  !  �  �  �  �  �  �    �  �  I  �  �  �  Y    *  �    �  z    �  �  �    8  �  w  �  H  o  s  Y  D  3  �  �  `  }  �    �  2  _  @  t  �  �    �  )  �  �  [  F  
�  �  �=�^5���ͼ�1��C���1���
�D���#�
���
�D��%   <u:�o;o=�P;�o<���;�`B<49X<��<�j<���<���<���<�`B<ě�<���=��w<�`B<�/=o<��=8Q�=+=�w=C�=t�=�P=��=#�
=<j=��=�w=0 �=0 �=L��=<j=<j=<j=H�9=Y�=]/=e`B=��=�7L=�\)=��=��=��
=��=��=�E�=Ƨ�=�
=>�u4228BN[grtspkg[NB>84.++.0<CDE@<0........�������������������AABEN[cgijihg^[NKFBA������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������FHO[hihh[OFFFFFFFFFFllnu������������zrnl{����������%#%)5BN[ituslgVNB5)%~���������~~~~~~~~~~"#%//07:92/+$#""""""ZRXboy������������gZ�����

 �����RNMTXamqwsmjaTRRRRRR���		���#0<?A<<20'$#)6BFGGB6)!�������������������ABBEO[`dhjkih[OMHDBA���������
����������������������������&'$���//1/,+/<=AHOTQQH<6//��������������������"!#,/:<HLUSJH<0/##""���������������������������������������������������������������������������������������������������������
#/9>FFC@:/���99;BDOT[\[XUOB999999Y[hjt���yth[YYYYYYYY)*5BDEB>65)��$)*%)'"!!���&'))57765)&&&&&&&&&&)).+) )+,,) ����).158CDB5-�55:BN[`c[UNB65555555������$((���{x{��������������������������������������*/6776/*����������������������������������������������
!#$%#
���)(),36A9766)))))))))
#$)*&#"
_\Z]akmnnona________::;=HTZacaYTIH@;::::``a^\amz~����}zomca`��������������������wz����������������{w��������������������jhnz���������|zsonjj������)�-�+� �����������������������Ľнݽ����ݽнĽ½½ĽĽĽĽĽĽĽ�Ź����������ŹŭŨŧŭŶŹŹŹŹŹŹŹŹ���
�������
�����������������������{ŇŔřŚŔŇ�{�r�y�{�{�{�{�{�{�{�{�{�{�'�(�3�?�<�:�3�,�'�!��$�'�'�'�'�'�'�'�'������
���
����������������������������������������������������������t¦§¦�t�m�t�t�t�t�t�t��������������t�z����������������������;�G�T�`�l�m�m�m�c�`�T�G�;�6�3�:�;�;�;�;����(�4�=�A�D�A�4�(��������������¦²¿��������¿²¦£�ּ�����ּԼ̼ͼּּּּּּּּּ���(�5�A�N�S�[�^�Z�N�A�5�(��� � ��	��*�6�>�7�6�*������(�*�*�*�*�*�*�*�*�G�T�`�m�����������|�y�`�T�P�;�5�0�2�=�G�Y�e�e�r�e�e�e�Y�X�V�Y�Y�Y�Y�Y�Y�Y�Y�Y�YD�D�EEEEED�D�D�D�D�D�D�D�D�D�D�D�D���5�N�g�������������s�g�N�A�5�(���
�������&�&���������������������H�T�a�k�m�r�n�m�a�T�K�H�=�D�H�H�H�H�H�H�/�6�;�H�P�T�`�T�H�;�/�"�����"�-�/�/�������������r�r�f�`�f�r�s������ÓàááååàÓÇÃ�ÂÇÎÓÓÓÓÓÓ�"�/�1�;�H�T�_�c�]�T�;�/��������������"�������ʾվҾʾ�������������������������D{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DwDsDzD{�ѿݿ�����%�*�*�(������ڿѿȿÿĿ��m�z�������������z�m�a�U�T�H�G�D�H�T�d�m����������������������������������������ààìõùÿùóìàÛÙÛààààààà���������������������������������������ûлһܻݻ�ܻлû�������������������#�������ܻлû��ûԻܻ��� ��ݽ������������ݽܽݽݽݽݽݽݽݽݾ��������������s�p�m�o�s�z�������"�/�0�;�D�H�I�H�B�;�/�"� ���!�"�"�"�"�a�n�zÃÇÊÇ�z�r�n�a�U�F�<�/�<�>�H�U�a�����׾�	��#�$��	��׾����������������ܻ��������������ܻػһܻܻܻܻܻܼ'�)�1�4�9�8�4�0�'�!���'�'�'�'�'�'�'�'��#�(�2�4�>�7�4�(�����������Ƴ���������
����������ƧƚƎ��zƂƎƚƳ��$�0�1�8�0�$����������������������������������������������������������	������	��������������������#�0�<�E�U�[�\�U�<�/�#���������������n�p�{ŀłŁ�{�n�c�b�`�_�b�e�n�n�n�n�n�n�:�F�S�_�l�v�l�l�_�S�F�:�-�!���!�'�7�:������������������������～�������ĽƽĽĽý����������������������T�`�h�m�y�}���������z�y�m�k�`�[�S�Q�Q�T�)�6�B�O�S�[�^�[�R�O�B�6�)�(� �#�)�)�)�)�"�$�.�5�4�.�"������"�"�"�"�"�"�"�"�����ʾ׾׾޾׾Ͼʾ����������������������~�����������������~�}�|�~�~�~�~�~�~�~�~ǔǡǭǱǰǭǣǡǔǈ�{�{�{ǂǈǎǔǔǔǔ�����ûлܻлû��������������������������h�t�w�t�o�j�h�^�[�Y�O�J�O�O�[�e�h�h�h�hŹ��������������������ŹŬŠŝŠšŭŶŹ²¿������� ����������¿²®¦£¡¤¦²��������������������������r�h�^�`�f�r�E7ECEPESEXEPECE=E7E.E7E7E7E7E7E7E7E7E7E7�ɺֺ�ݺݺٺֺɺ��������������������ɺ�  6 , 7 4 \ # a v S *  K F : � / _ \ = * ' Y ; J 3 L  2 ) ` X ( Z L x : . 6 N 2 ~ G s g 8 N W I e Y  S - { A t M i Z Q : f \ *      v  �  F  O  �  �  G  �  g     �  �  x  W  :  g  ,    �    �    �  �  �  G  �  T  [  #  �      �  u  �  �  �  {  �  �  �  �  /  �  �  �  �  �    <  ,    �  �  R  �  h  �      �  U  :  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  @R  	  �  �  [    ~  �  6  �  �    �  v  �    �  +  �  
.  E  0  3  5  6  6  4  -  &          �  �  �  �  �  �      >  a  �  �  �  �  �  �  �  �  �  �  �  y  `  C  !  �  �  �  �  �          �  �  �  o  9  �  �  �  `  +  �  �  e  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  h  ;    �  4  9  =  B  F  I  M  M  M  L  H  A  :  2  (        �  �  �  �  �                 �  �  �  �  �  ~  [  /    �  �  �  �  �  �  �  �  �  �  v  i  ]  P  C  5  (      �  �  �  �  �  �  �  ~  p  b  T  F  )  �  �  �  x  \  E  .       �  �  �  �  �  �    s  g  \  K  6  "     �   �   �   �   �   �  �  �  �  �  �  �  �  �    i  P  5    �  �  �  �  q  B      ?  d  �  �  �  �  �  �  �  �  g  =    �  F  �  D  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        �  �  �  �  �  h  D    �  �  �  �  R    �  �  �  W  �    Y    i  �  �  �  �  �  i  )  �  �  �  C  x  s  W  �  �  �  �  �  �  �  �  �  �  �  �    y  r  l  f  `  Z  T  �  �    (  >  H  D  0  	  �  �  `  !  �  �  9  �  t  �  _  �  �  �  �  �  ~  {  p  b  T  F  8  *      �  �  �  �  �  �  �  �  �  �  y  S  *  �  �  �  f  /  �  �  q  %  �  m    �    .  o  �  �  �  �  �  �  �  e  7  �  �    �  �    	  �  �    6  I  U  Y  T  D  1    �  �  p    �  \  �  Y  �      
    �  �  �  �  �  �  �  �  g  F  %  �  �  �  K    *        �  �  �  �  �  �  �  �  �  �  �  �  \  &  �  �  �  �  �  �    x  n  d  U  E  -    �  �  �  J     �      9  �  �  	              �  �  �  �  z  H  
  �  x  (  �  �  �  �  �  �  �  �  �  �  �  �  �  z  V    �  �  ]    �  w  y  z  u  l  ^  I  /    �  �  �  �  b  =    �  �  �  
  9    �  $  �  �  �      �  �  Q  �  ]    }  :  �  	�     �  �  �  �  �  y  ]  D  3  %    
  �  �  �  S  �  S  �  #  �  �  �  �  �  �  �  �  �  q  W  9    �  �  �  :  �  �  X  �  �  �  �  �  �  �  �  �  |  s  g  [  O  B    �  �  w  =    	  �  �  �  �  �  �  �  �  �  �  v  D    �  �  �  r  b  �  �  �      ,  7  0  "  
  �  �  �  L  �    �  0  q  �  �  �  �  �  �  �  {  c  G  *    �  �  �  �  c  L  S  Z  _  6  [  p  v  \  7    "  H  O  E  '     �  �  �  x  =  �  �  �  �  �  �  �  �  �  z  o  c  U  E  G  U  E  )    �  �  �  H  @  8  /  '        	    �  �  �  �  �  �  �  �  �  �  o  l  j  e  Z  P  B  2  #       �  �  �  �  �  p  R  6    s  l  a  S  9    �  �  �  �  W    �  r    o  �    Q  �  S  Y  Q  5    �  �  n  0  �  �  E  �  u    �    i  �  r  g  �  �    "  4  A  C  7  #    �  �  �  y  G    ]  �  =  3  %    
  �  �  �  �  �  �  �    	                �  �  �  �  �  �      �  �  �  �  �  �  i  B     �   �   w  �  �  �  �  �  ~  l  ^  A  %    �  �  �  m  9    �  �  �  `  ^  [  X  U  S  P  C  0      �  �  �  �  �  �  �  l  W  |  z  w  y  |  }  |  n  Z  C  *    �  �  �  _  #  �  �  ]  �  �  �  ~  n  [  E  .  "      �  �  �  r  L  %   �   �   �         �  �  �  f  8  
  �  �  �  �  Z    �  �  H  �    �  �  �  �  �  b  D  &    �  �  �  �  s  Y  @  9  7  P  l  1  *    �  �  �  �  �  p  M    �  �  `    �  R  �      _  H  0        �  �  �  �  l  I  !  �  �  �  ~  V  -    @  !    �  �  �  �  �  �  �  m  Z  D  +    �  �  �  �  �  t  r  n  h  X  D  ,    �  �  �  �  z  U    �  y     �   b  �  �  w  Z  ;    �  �  �  �  `  3    �  �  q  >  	  �  �  �  �  �  �  �  �  �  �  �  �  s  d  N  9  #    �  �  �  �  �  �     �  �  �  �  �  �  �  b  ;    �  �  �  T  !  �  �  �  �  �  �  �  v  l  b  X  N  K  N  R  V  Z  ^  a  e  i  m  �  �  �  �      "  	  �  �  �  �  i  %  �  \  �  ~  )  �  �  �  }  Y  .  �  �  �  d  +  �  �  u  .  �  �  T    �  �  �  �  �  �  �  �  �  �  �  j  I    �  �  �  �  �  ~  t  i  [     �  �  �  �  �  z  Q  (  �  �  �  �  [  7        �  F  
�  
�  
�  
�  
�  
c  
<  
  	�  	�  	3  �  )  �  �  c  �    �  
�  
#  	�  	�  	�  	�  	\  	+  �  �  H  �  �  -  �    q  �  �    �  �  �  �  �  d  :    �  �  �  Y    �  �  L  	  �  }  .  �  �  q  9      
�  
�  
�  
?  	�  	�  	  �  �  -  <  =  3  