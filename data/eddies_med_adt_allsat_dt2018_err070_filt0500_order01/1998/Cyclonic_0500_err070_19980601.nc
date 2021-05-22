CDF       
      obs    >   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��Q��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��1   max       Pam}      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��v�   max       <u      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�=p��
   max       @E�\(��     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vzz�G�     	�  *D   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @3         max       @P�           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @�@          �  4p   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �O�   max       %         �  5h   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B,�b      �  6`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��L   max       B-:N      �  7X   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�   max       C���      �  8P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�	   max       C��      �  9H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          =      �  :@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  ;8   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  <0   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��1   max       P\C�      �  =(   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?����l�D   max       ?ϱ[W>�7      �  >    speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       <e`B      �  ?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @E�          	�  @   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vzz�G�     	�  I�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @3         max       @P�           |  Sp   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @��          �  S�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E   max         E      �  T�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�6��C-   max       ?Ϫ͞��&     `  U�      
                                 (         
      $                           3         
            
            4            
      3   <   
   
                  '         (   	            5   O��N3l�N=E�N��MN���O't�N�,NT��N�O@/�O�\�N�PP]!O���N�WO8�/NH�PGҰOU,]OCC�O�UQN��mO��O��O�]�O�8VPam}NT�WO)|YN��GO+o%N���O�N���N��OAN{N`�LO��O��OJ��P!%N��DO�P)��P ��N�&�N-O��OAx�OyN�O���N�%�O��9M��1O�	O�w1N��-N�RIO��O��O�=�O�<u<t�:�o:�o%   %   ��o��`B�t��D���T���e`B��C����㼣�
��1��1��1��1��9X��9X��9X��j��j�ě���h��h�����o�\)�t���w��w��w�#�
�#�
�',1�,1�0 Ž0 Ž0 Ž0 Ž49X�49X�8Q�D���D���H�9�P�`�T���T���T���Y��]/�]/�aG��q���q���� Ž�v������
#-20&
������$#!��������������������BIUbicbZUMIEBABBBBBB��������������������mnwz������������zwqm�����	
	�������������������������pz���������������zpp#/<HY_]URH><8/.*#����������������������������������������Uanzx����������znaVU2:N[bgt}�����xq[N912/6BEOOOB6*//////////*/3<Uamnvy{naUH</,(*���������������������#@Lanqmuzzvb<0���{~�������������{qvx{!#/<HUY\ZUH<7/#'('#! )6BOV[^XV][OB74.&$ ����������������������)BNW[UNB)��������
������mz�������������zmjimMO[egt�������tod[QMMz���������������tgiz;;>GHRT[XVTQH;;;;;;;������������������z�CHOUWajmaUHFCCCCCCCC�����������������������������������������������

���������ABO[htxtjhb[YOKB@:AA��������v���������������
%#!
������
#(# 
�����������������������
#+*#
����������������������5Ngt�������og[NJHB95�������������������������������������������������������Um���������������mWU7<FHMRQHB<:776777777./5<@A</-*..........�����������������������������������������������������������Uanz���������zjaUPPU������������������������	 ���������!GILNRU[bfkpmpnibUIBG )59BKOQPTNB5)&" Y[agtv�������xtpgbWY��������������������BNWSNMIB)���]ggt����������~tge]]##&/<FUalUN</##���(),))�����ìà×Ä�}ÅÌÕàìù��������������ùì��ۺֺѺֺ������������������s�r�g�s�����������s�s�s�s�s�s�s�s�s�s�ʼļ��ļʼּؼ�������ּʼʼʼʼʼ��#���
����������
���#�/�<�?�?�<�/�#��������������������������� ��������������	�����������������	��"�$�/�/�*�"��Ŀÿ��Ŀѿݿ����ݿѿĿĿĿĿĿĿĿ����������������������������
���6�/�-�,�-�6�B�O�U�[�g�t�v�t�k�h�[�O�B�6�������ڽ׽ݾ��(�4�C�M�M�A�:�4�(�ŭŨŠŘŔőŔŠŭűŹż������Źŭŭŭŭ�s�0���(�N�]�g�s���������������������s�����v�q�`�Z�X�T�G�T�`�y�����������������;�7�2�;�G�H�I�N�M�H�;�;�;�;�;�;�;�;�;�;�/�)�"������"�/�;�7�>�K�N�I�J�H�;�/�������������������������������������������������s�g�A�5�A�Z�����������������������ݿԿ˿ͿѿԿݿ������'�2�(�����
�� �����#�/�=�>�C�H�M�H�<�/�#��
�F�>�6�@�G�S�i�������������������x�_�S�F�T�R�G�A�9�7�;�;�D�G�P�T�U�Y�_�`�f�f�`�T�ʾþ����������ʾ׾�����������׾��Z�P�N�J�N�P�Z�g�g�s���������������s�g�Z�����������	�"�/�;�H�T�V�U�R�T�P�=�"�	��¦�y�}£¦²¿��������������¿²¦�(��	��
����g���������j�a�^�_�N�A�(������������������������ѿпĿ¿����������Ŀѿݿ������ݿѾM�C�A�@�A�E�M�Z�d�d�^�Z�M�M�M�M�M�M�M�M���������������������������������I�H�<�4�2�<�I�U�Z�`�b�d�b�U�I�I�I�I�I�I������߾ھ�޾��������� �#����	�������������������������������������������g�g�]�g�s���������s�g�g�g�g�g�g�g�g�g�g������������!�+�-�2�6�:�B�<�:�1�!��S�G�L�S�\�_�d�l�u�v�l�_�S�S�S�S�S�S�S�S�S�:�!���"�#�-�:�F�V�o�����������x�_�S���������������ĿȿѿԿݿ��ݿٿѿĿ���àÖÓÉÇÁ�ÇÎÓÞåìîùþÿùìà����������������)�2�N�S�N�:�)����������	�������$�*�/�4�6�*������Y�U�L�I�C�@�8�8�@�L�Y�e�o�r�v�x�w�t�r�Y�f�}������������$�(�%�����ּ����r�f������������������������$�+�+�������D�D|D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�FFFFFF$F+F*F$FFFFFFFFFFF�6�+�)�$�)�2�6�B�O�[�h�q�n�h�b�[�O�B�6�6ĦĢĚčā�t�t�|āčĚĦĳĿ��������ĿĦƁ�u�h�]�R�P�Y�h�uƁƉƎƧƩƭƪƧƚƎƁ�����������ùϹ����'�2�'�!�����Ϲ�����������������������������������������ͼ4�'��	� ����'�4�@�M�Y�e�o�n�f�Y�@�4�ܻۻлʻлܻ���������ܻܻܻܻܻܻܻܼ����ܻл̻л׻ܻ������'�*�$����[�Q�L�O�W�h�tāčĦĬĮĳīĦĚā�t�h�[ĳıĳĵľĿ������������������������Ŀĳ�~�r�p�r�~�~���������������������������~ÓÀ�n�^�>�@�H�a�n�zÇÑæëòýùìàÓ�<�/�/�'�'�+�/�;�<�H�U�\�`�W�U�M�I�H�<�<E�E{EuEtE~E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������������������!�$�-�+�!���� & T S @ W ; d 3 ^ 9 W j P n ` d - _ U 4 \ q k , I / E o H > 1 K p j a M : B G _ 8 Z > Z Q S O 5 F E T 8 Y m F F � H p F K ;  a  T  N  �  5  w    a  >  �  u  �  �  �  X  �  S    �  �  y  L  �  ;  �  q  �  �  t  �  o  �  �      �  l  b  _  �  �    d  �  �  �  +  E  �    �  �  �  1  |      7  %  U  J  F��`B%   �o��o�o�49X�#�
�#�
��/�o����t��ixս�㼴9X�����ixսC��\)�L�ͼ�h�,1�\)�#�
�@������\)�',1�'<j�u�D���8Q�e`B�0 Ž�j�]/��o���P�T���]/��vɽ����]/�aG���C��e`B������w�����Q�q���u����}󶽍O߽��罣�
�O߽�l�B�DB�	B�B'"BB��B_A���B,�bBjB��B!�BL,B{|B	6B%B{0B �WB%��B)�WB�JBA�B+��Bm�BqoB �B	��B�eA��MBPAB�kB��B�B�8B �B[B#FWB$�B�IB�:B��B	��B�QB!�aB,��B�oB��ByB�B�\B-�B�B
�mBo�B�B'�SBF<B	��B�B�qB
C�Bb�B%HBvRB��BBOB&ÄB��B�,A��LB-:NB��B�	B"#�BAB�bB`PB��B�sB �MB&;(B)��BúB1�B,BBB�B 4B	�-BA)A�~,B��B�GB�nB�B6RBǘB�gB#$�B$NB@�BJB�B	?�Be�B":EB,��B�B��B@`B��B@�B?!B<�B
��B?�B�rB'B�BB�B
AGB�B�B
�mBB>B5WA̡�@I'�AE-A5�A��	AЯkA�(bA{��A���A��A4f�A��QA���AnSA�47A���@���A��cA�A��a@�.�AfۭAR��A�&A��{A��TA��"A�1QA{+�A>~B1bA�9AY��A�0�A�m�@f�m@�E�@���Ax��A�q/A��XA�Y�?���A x�A�#C��C���A�7A�E�B4�?�B:�@��k@�wS@�i^A�AhA�h@t�AȘ.A��'C�&�A	�5A�d@G��AD#bA �#A�	�A��A��A{0-A��A��kA4�?A�y�A�c�Ap�A�j�A�z.@��A�{;A��NA�vM@�nQAd��AS�A���A�aA�HxA�UA�E�A{v�A=��B��A��AY�A�=�A��@c��@��N@�>?Ax�_A�4IA���A�>?��A9XA怳C�՚C��A�g�A��KB=>�	B\*@х�@��h@��|A�7qA��@1�AȂ�Aé�C�'"A
�2      
         	                        )               %                           3                     
            4             
      3   =            	            '         )   	            5                                    !      9   #            ;         !      !      !      1                                 '         +         5   '                  #                        #                                                /   !            -                     !      1                                 '                  -   !                  #                        #         O���N3l�N=E�N��MN���O't�N��NT��Nr�.O�Oa�N�P7 �O�%�N�WN�Y[NH�P.KROU,]OCC�O
F6N��mN�?�O��O�]�O�۹P\C�NT�WO)|YN��GO+o%N]�eO�N���N��N۔�N`�LO�� O��OJ��O�FbN��DO�P��Oʿ�N�&�N-N�H�OAx�OyN�O���N�%�O���M��1O�	OQp�N��-N�RIO��N��pN��O�  �  �    0  �  �  �  Y  N  �  [    ]  �  �  �  �  �  	  �  y  �    �      �  �  }  2  k      �  z  r  �  p     M  ]  �  �  9  	  �  %  �  �  <  (  �  �  4    G  N  &  <  �  	�  �<e`B<t�:�o:�o%   %   ���
��`B�u�e`B��o�e`B���
���
���
��j��1��j��1��9X��󶼴9X�o��j�ě����������o�\)����w��w��w�49X�#�
�49X�,1�,1�<j�0 Ž0 Ž<j�T���49X�8Q�L�ͽD���H�9�P�`�T���]/�T���Y��m�h�]/�aG��q����%������v�����
#,1/%
�������$#!��������������������BIUbicbZUMIEBABBBBBB��������������������mnwz������������zwqm������			������������������������������������������������#/<HRUWUUOH<0/+%#����������������������������������������ant�����������naYWWa3;N[_cgt����wp[N<533/6BEOOOB6*//////////+2<HU^anqrna_URH<2/+��������������������#EUbnwwtbF<0
���{~�������������{qvx{!#/<HUY\ZUH<7/#'('#!.6BOQTWYWOHB?;63/,..�������������������������������������
������mz�������������zmjim[gtz�������tsg[SNNQ[����������������thj�;;>GHRT[XVTQH;;;;;;;������������������z�CHOUWajmaUHFCCCCCCCC�����������������������������������������������

���������ABO[htxtjhb[YOKB@:AA��������v�����������������	����������
#(# 
�����������������������
#+*#
����������������������FOgt���������g[OLJJF�����������������������������������������������
�������lz���������������~ll7<FHMRQHB<:776777777./5<@A</-*..........�����������������������������������������������������������Uanz���������zjaUPPU�������������������������������������!GILNRU[bfkpmpnibUIBG#)46BJMNMBA5)($!#Y[agtv�������xtpgbWY��������������������BNWSNMIB)���fgmt�����������tpgff ##/<EHJJHB<:/#"!   ���(),))�����àÚÆ�ÇÎÖàìù��������������ùìà��ۺֺѺֺ������������������s�r�g�s�����������s�s�s�s�s�s�s�s�s�s�ʼļ��ļʼּؼ�������ּʼʼʼʼʼ��#���
����������
���#�/�<�?�?�<�/�#��������������������������� ������������"���	���������	��"�#�,�$�"�"�"�"�"�"�Ŀÿ��Ŀѿݿ����ݿѿĿĿĿĿĿĿĿ�������������������������������6�1�.�-�/�6�>�B�M�O�[�c�t�h�g�[�O�K�B�6�(��������������(�4�<�G�C�<�4�(ŭŨŠŘŔőŔŠŭűŹż������Źŭŭŭŭ�5�!�(�N�Z�j�s�����������������������s�5�����w�r�m�`�[�Z�L�T�y�������������������;�7�2�;�G�H�I�N�M�H�;�;�;�;�;�;�;�;�;�;�;�/�"� ��"�&��"�/�2�7�;�F�H�L�F�H�H�;�����������������������������������������������s�e�g�v������������������������������ݿԿ˿ͿѿԿݿ������'�2�(�����
�� �����#�/�=�>�C�H�M�H�<�/�#��
�S�G�H�O�S�_�l�x�������������}�x�l�_�S�S�T�R�G�A�9�7�;�;�D�G�P�T�U�Y�_�`�f�f�`�T�׾Ծʾ¾��ʾ׾����۾׾׾׾׾׾׾׾��Z�P�N�J�N�P�Z�g�g�s���������������s�g�Z�����������	�"�/�;�H�T�V�U�R�T�P�=�"�	��¦©²¿������������¿²¦��
�������Z�s���������i�`�^�N�A�������������������������ѿпĿ¿����������Ŀѿݿ������ݿѾM�C�A�@�A�E�M�Z�d�d�^�Z�M�M�M�M�M�M�M�M���������������������������������U�S�I�<�7�4�<�I�U�X�]�X�U�U�U�U�U�U�U�U������߾ھ�޾��������� �#����	�������������������������������������������g�g�]�g�s���������s�g�g�g�g�g�g�g�g�g�g�� ����������!�-�5�4�-�,�!������S�G�L�S�\�_�d�l�u�v�l�_�S�S�S�S�S�S�S�S�_�F�.�!���$�%�-�:�F�S�m�����������x�_���������������ĿȿѿԿݿ��ݿٿѿĿ���àÖÓÉÇÁ�ÇÎÓÞåìîùþÿùìà������������������)�-�8�M�H�5�)�������	�������$�*�/�4�6�*������Y�U�L�I�C�@�8�8�@�L�Y�e�o�r�v�x�w�t�r�Y������������������� �%�#�����ּ����������������������������&�%���
����D�D|D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�FFFFFF$F+F*F$FFFFFFFFFFF�6�-�)�'�)�5�6�B�O�[�h�i�i�h�`�[�O�B�6�6ĦĢĚčā�t�t�|āčĚĦĳĿ��������ĿĦƁ�u�h�]�R�P�Y�h�uƁƉƎƧƩƭƪƧƚƎƁ�����������ùϹ����'�2�'�!�����Ϲ�����������������������������������������ͼ@�4�'����
��'�4�@�M�Y�d�n�l�f�Y�M�@�ܻۻлʻлܻ���������ܻܻܻܻܻܻܻܼ����ܻл̻л׻ܻ������'�*�$����[�T�O�O�Z�h�tāĚĦĦīħĦĚčā�t�h�[ĳıĳĵľĿ������������������������Ŀĳ�~�r�p�r�~�~���������������������������~ÓÀ�n�^�>�@�H�a�n�zÇÑæëòýùìàÓ�<�:�/�+�*�-�/�<�H�H�I�U�[�U�S�J�H�>�<�<E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������������������!�$�-�+�!���� & T S @ W ; S 3 \ 1 K j K l ` g - S U 4 @ q 1 , I 1 F o H > 1 < p j a & : B G _ + Z > R = S O , F E T 8 O m F @ � H p C 4 ;  H  T  N  �  5  w  �  a  �  W  �  �  K  �  X  U  S  \  �  �  ;  L  �  ;  �    �  �  t  �  o  {  �      �  l  C  _  �  �    d  "  �  �  +    �    �  �  *  1  |  �    7  %  �    F  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  �  �  �  �  �  �  �  �  �  �  �  �  �  s  2  �  �  �  |  v  �  �  �  �  �  �  �  �  p  Q  /    �  �  �  {  T  ,    �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  r  j  b  Z  0  '          �  �  �  �  �  �  �  �  �  r  z  �  �  �  �  �  �  �  �  �  �  l  I  $  �  �  �  o  <  	  �  �  �  �  �  �  �  �  �  �  �  �  {  q  k  g  d  `  ^  d  h  i  n  u  �  �  �  �  �  �  �  �  �  �  �  p  ]  J  6  !      �  �  Y  V  R  N  K  G  C  =  5  .  &           �   �   �   �   �  �    /  =  G  L  M  K  E  D  F  ?  '    �  /  �  �  F  �  �  �  �  �  �  �  �  q  [  ?    �  �  i    �  }  5  �  �  I  S  Y  [  Z  T  H  7  %    �  �  �  �  d  2  �  �  s      w  o  g  `  X  S  N  H  C  9  ,        �  �  �  �  w  .  \  W  ?    �  �  �  }  R  %  �  �  �  N    �  s    �  u  �  {  s  T  .    �  �  I  7    �  �  �  �  a  �  r  �  �  �  �  �  �  �  |  u  o  h  b  \  V  Q  K  E  ?  :  4  .  S  g  y  }    y  v  {  {  r  ^  ?    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  S  '  �  �  �  i  7    �  �  _  ;  �  �  j  R  :    �  �  �  b  $  �  �  �  d  C  	  �   �  	  �  �  �  �  �  �  �  �  �  �  i  O  4    �  �  �  �  N  �  �  �    g  P  <  -      �  �  �  x  W  A  ;  4  %    �  �  	  0  \  s  y  t  m  g  Y  E    �  �  M  �  O  �  �  �  �  �  �  �  �  �  u  g  Q  9       �  �  �  �  �  �  e  �  y  k  \  K  3  "  '  �    �  �  �  �  c     �  9  �  9  �  �  �  �  �  �  �  �  �  �  u  c  N  7    �  �  �  �  �      �  �  �  �  �  p  N  1    �  �  �  �  t  R  4      �        �  �  �  �  �  |  U  *  �  �  �  C  6    �  |  �  �  �  �  �  t  `  I  -  ?    �  �  \    �  �  a  �  8  �  �  ~  o  b  [  S  K  D  =  5  .    	  �  �  �  �  �  �  }  p  a  P  <  &    �  �  �  �  �  �  p  O  *    �  �  H  2  $      �  �  �  �  �  �  �  �  �  u  i  _  Y  T  Q  N  k  Z  J  9  &    �  �  �  �  �  �  �  z  g  U  @  %  	   �              �  �  �  �  �  �  �  h  H  (  �  �  R      i  M  $  �  �  �  �  �  �  �  �  �  ^  ,  �  �  X  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  n  ]  3  	  z  q  h  _  G  -      �  �  �  �  �  �  �  t  _  I  2    S  R  Z  e  n  r  q  m  g  ^  S  F  7  $    �  �  �  C  �  �  �  �  �  �  �  �  �  �  �  �    }  �  �  �  �  �  �  �  k  p  o  b  D     �  �  �  Z    �  |    �    H  l  �  �       �  �  �  �  �  �  d  =    �  �  �  �  �  �  y  ^  B  M  >  +    �  �  �  �  ]  0  �  �  �  .  �  �    �  �  &  9  J  [  N  =  0  ,    
  �  �  �  �  �  �  c    �  �  �  �  �  �  �  x  l  _  P  @  ,    �  �  �  �  p  K  &    �  �  �  x  c  K  /    �  �  �  �  m  N  2    �  �  �  i  *  �  8  4  "    �  �  �  �  r  F    �  �  @  �  �    T     �  	  	  	  �  �  �  l  7    �  q    �  L  �  J  �  s  �  �  �  �  f  J  8  #  	  �  �  �  �  �  �  �  �  �  �  �  �  %  =  T  \  a  [  T  K  =  *    �  �  �  ]  *  �  �  �  O  �  �  �  �  t  c  P  <  $    �  �  �  U  "  �  �    �  "  �  �  �  {  f  Q  :  "  	  �  �  �  �  e  :     �   �   �   �  <  4  ,  *  %      �  �  �  �  d  =    �  �  u    �    (       	  �  �  �  �  Y  #  �  �  O     �  S  �  �  �   �  �  �  z  f  Q  I  H  H  9  (  *  -  2  8  A  L  W  s  .  v  �  �  �  �  b  &  �  �  j  &  �  �  U    �  M  �  	  T  7  4  A  M  P  =  )  	  �  �  �  c  ;    �  �  �  �  b  <      �  �  �  �  �  �  �  k  T  =  )  %  !           �   �  �    G  @  0       �  �  �  �  Z    �  A  �  �  �  �  �  N  @  3  )      �  �  �  �  �  �  �  f  G  '    �  �  �  &      �  �  �  �  �  �  a  =    �  �  �  ^  !  �  �    <  )    �  �  �  �  �  �  �  b  3  �  �  L  �  F  �  �  %  �  �  �  �  �  �  �  �  ~  c  @    �  �  ~  M    �  �    �  �  	I  	d  	�  	�  	�  	�  	�  	m  	.  �  W  �    w  �  �  �  q  �  �  �  �  �  �  |  _  5    �  �  S    �  �  C  �  �  �