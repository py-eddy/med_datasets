CDF       
      obs    Q   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��t�k     D  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       PcP�     D  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���-   max       <�o     D   4   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?:�G�{   max       @F������     �  !x   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��G�z�    max       @v��\)     �  .    effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @P            �  :�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�7          D  ;l   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �   max       <o     D  <�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�H   max       B4
     D  =�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��A   max       B4�_     D  ?8   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >~�=   max       C���     D  @|   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >L��   max       C���     D  A�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          V     D  C   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9     D  DH   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /     D  E�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P#��     D  F�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�,<�쿲   max       ?Թ#��w�     D  H   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���-   max       <49X     D  IX   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?:�G�{   max       @F������     �  J�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v��G�{     �  WD   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @P            �  c�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�3          D  d�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�     D  e�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�*�0�   max       ?Թ#��w�     P  g         	         	                           	   	             &                           G      
                     	                        /       V      0            	   
         )            
                                                      (   O��NZy
N�,
M� M��OO�qN��{Nk��O�!dOz=O��N�.�N]�~O��fN��NbjdO�%O���N�O�^�N�M�I�O�!O�
O�N��N���Ow�PcP�N�\�N�lN�"�O�L|NQGND-�O&�O/fWOfO��|N��N!��N��OZ�O��Nm	^O{=�O�ƺP6��Nu 6P��N��~N0;NC>�N�-N��N��aO��N��zOXglOkqN�5�O�'N���N�Z�O'GyN�4;N���N��xO4�N�5�N��>N�OG�N��OJk�NJSNu�O�pN '�OaтO���<�o<49X;ě�;D��;o$�  �o���
��`B�o�49X�D���T���e`B�e`B�e`B�u�u��o��C���C���t����㼣�
���
��1��1��9X��9X�ě��ě��ě����ͼ��ͼ�����/��/��/��`B��`B��h��h��h�������o�o�o�+�+�\)�\)�t���P��w��w�#�
�'''',1�,1�0 Ž49X�49X�@��@��H�9�T���T���]/�ixսq���q���y�#��%��O߽��P���-����
#.2+#
�������wz�������zxnwwwwwwww��������������������JOW[_\e_[YWOJJJJJJJJjnrz����znjjjjjjjjjj�������


�����������������������������������������������
#-#($�����������������������������������������&)6BCILKB=61)
#%#!
����������������~}������� ������������
#(-)#
����������{vlh__bhn�5;HLTmrpqnpmTKJC;335����������������������������������������#./05//#��������������������*4GO[hosszz|{h[O6(%*mt������������xtokim��������������������aanxz{�zwnida]aaaaaaght�������~thhgegggg#/<HLPPHE</$#
�����/0���������_almvz�����zwma`____)5@BEB95)JNQ[gtuwtqig[NFFJJJJWdt������������tg[WW����������������������������������������>BN[gtx�����thg[NB:>�����

������*/01;<HQU[ZYUMH</)**��������������������259BJNTZ[][NB652/-22����������������������������������������ku���������������ukSTWamnz~zyvrmjaXTQQS|�����������||||||||-0<IU`bloqsnbUI<50.-anz����������znia^^a������� ����������Y[htt���tmh[W[`[YYYY��)5BUjvvjNB�����������������������o{�����{rooooooooooou{�������{}{uuuuuuuu����������������agt�����ytgca]aaaaaa������������	
#0<IMRLI<30#
	����	

	���������������������������� �����������������������������������������������������������������������������������������������������������������������������������������������������)*05=BJB5)$Yglt�������}ztgd[SOY
#000240#
��������������������KNQ[_c[VONMGKKKKKKKK�����������������������������	�����KOX[ahjw~}tvtjgYOKIK#0260(#!����������������������������������������������������������$')(%�����#/H[UHE>64-#��òìíïîðù�����������������������������������������������������������������H�G�B�C�H�U�a�j�n�s�z��z�n�a�U�H�H�H�H�;�1�/�,�/�;�H�T�U�T�H�A�;�;�;�;�;�;�;�;�U�P�H�F�F�H�U�U�W�V�U�U�U�U�U�U�U�U�U�U�M�F�4�/�/�&�(�4�A�L�Z�g�����w�o�f�Z�M�t�s�h�t¦©²¶¸²¦�����������������������Ľ��������{�}���������Ľݽ�ݽнн�Խ�������������������$�0�:�>�=�5�0�$��àÓÇ�w�k�j�n�zÀÇÓàìùÿ��ýùìà�#�������#�/�<�F�H�O�H�D�?�<�/�#�#�ݽԽѽٽݽ����������ݽݽݽݽݽݽݽݿ��ݿؿͿȿοѿݿ������������������%�(�5�A�N�N�N�L�H�A�5�(����ŭŪŬŭŴŹ������������Źůŭŭŭŭŭŭ�l�w�y�������Ľн��ݽн����������y�t�l������&�5�N�g�s�������s�g�N�4�(���z�y�m�i�m�z���������z�z�z�z�z�z�z�z�z�z�ѿ��������ʿѿݿ���������������A�>�5�/�(�(�(�5�A�A�G�G�A�A�A�A�A�A�A�A�*�(������*�/�6�7�6�*�*�*�*�*�*�*�*��	�����������.�;�G�`�t�q�j�R�;�.�ŔŊŉŔśŠŪŭ����������������ŹŭŠŔŔŊŇ�{�u�r�{ńŇŔŠŬŭŰűůŭŠŔŔ�����������������������������������������������������������ƾʾӾʾʾ������������O�6�/�)�,�.�3�6�B�O�_�h�u�t�h�c�f�d�[�O�Z�^�t�~������.�4�3�����ּ�����r�Z�"����	��	����"�-�/�5�3�/�"�"�"�"�ѿп˿Ͽѿѿݿ������ݿѿѿѿѿѿѿ��������������������ĿƿͿ̿Ŀ����������	����������������������#�,�2�@�/� ��	��������y�������������������������������������������	��������	���������������������������������������������m�d�a�T�H�G�D�H�J�T�a�m�r�z�}���{�z�m�m�"������	��"�/�;�T�[�a�e�d�]�T�H�;�"������#�*�6�;�C�E�N�M�C�A�6�*����x�w�l�j�_�]�_�l�w�x�����x�x�x�x�x�x�x�x�	����������������	���"�$�0�/�/�"��	�����������������������������žž�������������ƾƳƳƱ�������������������������ƧƟƚƎƋƆƎƚƧƨƲƭƧƧƧƧƧƧƧƧ�(���
�
����(�4�A�M�X�\�^�X�M�A�4�(���������3�@�L�Y�[�L�@�0�'����������ܼ��1�M�f�������ʼ������Y�@��àÖÛààéìóòù����ùòìæàààà�������������������������"�6�7�)��������������������������������������'�!�&�'�4�@�@�F�@�4�'�'�'�'�'�'�'�'�'�'�5�-�(��(�5�A�N�R�N�A�?�5�5�5�5�5�5�5�5��Ʒ����������������������������������������������� � ���������������F1F=FJFVFZFcFoFoFoFcFVFSFJF=F3F1F/F1F1F1����������������������������������������D�D�D�D�D�D�EEEE*E,E*EEEEEED�D��T�G�;�.�+�"�����"�.�;�D�T�a�k�i�`�TßÞÙÙàù����������������������ùìß�t�n�h�d�h�o�tāčĚĦıĦĢĚđčā�t�t��������ĿĴĿ����������������������������������
��#�)�0�#��
�����������������U�J�R�U�b�j�n�{Łń�{�z�n�b�U�U�U�U�U�U�L�F�D�E�L�M�Y�e�r�s�~������x�r�e�Y�O�L�"�����"�,�.�/�;�G�H�M�G�;�.�"�"�"�"������������������������������������ā�x�āčĔĚĢĦĪĬĪĦğĚčāāāāĚđčĉĆčĎĚĞĳĹĿ��������ĿĳĦĚ������������������%�%���������������������������������������ìãàÙàìóù������ùìììììììì�#��
�������
��#�/�;�H�U�Y�U�N�H�<�/�#���������}�y�l�a�l�y���������������������ù��������������ùϹܹ���������ܹϹû!����!�-�7�:�;�:�0�-�!�!�!�!�!�!�!�!�S�M�Q�S�_�l�x�������������x�l�_�S�S�S�S��	�	����(�5�A�N�Z�_�d�e�d�Z�5�(���������������������������������������������������������!�.�:�G�H�G�?�:�.�!�E�E�E�E�E�EuEmEnEsE�E�E�E�E�E�E�E�E�E�E� L < 6 a } < X j ^ ) M # # 4 , q @ n j 5 i q . G . E 8 8 _ X L 6 K I J g ( 1 ) K ~ U g Z Z  b l d V I ( O m ) P Z T b 0 T L I Y 5 A T e > w v x N N H U x = v  �    �  n  �  9  Y  �  9  �  �  �  u    d  2  �  �  N  �  *  ]  b  6    2  8  �  �  �  x  �  �  �  w  k  z  �  �    u  '  r  A  d  w  �  �  p  �  �  �  �  M  \  L  �  �  j  B  �  �    5  �  �  ~  �  �  �  �  �    G  �  6  �  �  �  .  C  �  ��e`B<o�o�o��o�t���t��#�
�������'�������\)��j��9X�@����ͼ���]/��1��9X�H�9�#�
�0 Ž+�o�,1��j��/�+�\)�Y���/����P�,1�t��Y��#�
�+�@��#�
�8Q�+���w��%��\)���T�#�
�#�
��w�49X�<j�m�h�<j���
�]/�T���]/�L�ͽL�ͽ@��y�#�P�`�u�L�ͽm�h�T���u�ixս����%�� Ž�o��C���vɽ�hs��l�����B��B��B!(QBBIB�{B�QB �OB#ЀBo�B��B��B. BK�B:�B?B)pA�HB B�B�(B@�B�vB
�tB]B�hBQ�B
�B-��A��B��B�mB
��B
ޝB��B	2cBIbB�B��B��B!_�B!U�B4
A�:B ��B&�Bw�B��B�B�Bk
B(ЏB))�B��B	�%BH�B%��BިB��B��B�B�_B�@B%HB!{iB�B�lB�B	��B$��B�8B��B�B �BoB%$�BB_�BG	B	�BF�B�AB��B!L�B��BL�B�*B�iB �vB$@ Bv�BFB�&B?�B5B@�B��B)A{A��ABbB=sB��BDnB�"B
��B,$B��B@ BƏB-�XA��B��B�MB?qB1B 5�B	8B7�B��B��B��B!��B!^wB4�_A�x B ��B&�5B?0B6B�BB9B�
B(�[B)9^BA�B	��B@B&-B�WBJB�$B�B��B;�BBB!��B?�B�DB��B	>B$F�B�RB?�B�B@rBU�B%@�B1�BB�B�FB§B<�A�p�AЮ7A�JyA��hA�8A>z6A� �A/��A#�1B	M�A��%A���A,,�A�`,A�u�A��A")A��A��A~+TA���A�_+Aaa�A�-A��A��{AM��A�[bA��A�1A}MAv�A���A��@�,AY��A���A���A���A��I@��A��AL��B�gB^AA8ɜ?�mK@��'A�y�A��A�	�@�v�A�ŋB3TAӣ�C���A�MC�U�Ad,�A�YAݤ�A�$KA�5)A�?䫁AaCqA��\A���A�B@��A� uA���A���Aޗ>~�=@n��@�/�A�|WA���Ah;C�/A��8AИqAŅ2A�}*A��-A<�tA�zHA1aA! �B	�oA�i�A�
,A,�A�qWA�}`A�}{A#VA�9A�s�A~��A�}<A��Ab�pA���A�|!A��MAM(A���A�A��A}
�Au�A�O{A��@�.AY3�A���A��OA��mB  �@��=A���ALK�BKB�A8�Z?j�c@��FA��A���A���@���A��B�)A�E�C���A��\C�W�Ab�A͔�A�[�A��SA�A�$?�h�A`��A�j=A�6<A�#@�E�AЈ#A�{OA��A��>L��@mWG@��A�� A��&A~C�&         	         
                           
   	   !         &                           G      
                     
                        0       V      1            	   
         )            
   	                        	                           (                              '                           !               %                  9            '                                             5      '                                                                                             #                                                      !               %                  '            '                                             /      %                                                                                             #O.�NZy
N�,
M� M��OO�qN��{Nk��O$�eO+O�TfNS1NN]�~OB?�N��NbjdOPsDO���N�O(��N�M�I�O�TOxV�N��LN8U�Ns��Ow�O��?N�\�N�lN��O�L|NQGND-�N��O��OfO�-�N���N!��N�p�OZ�O��Nm	^O@IO�ƺP#��Nu 6O��N��~N0;NC>�N�-N��N^_yO��N�f�OXglOkqN�)]O�'N���N�Z�O'GyN�4;N���N��xO4�N�5�N�CN�OG�N��N^��NJSNu�O
[N '�OaтO���  �  �  �  �    2    �  �  �    ^  b  -  z  !  �  t       �  _  Q  q  *    Y  �  �  �  p  z  {  �  �  *  j  3  �    �  �  �  �  �    �  X  �  �  1  �  �  I  ;  �  O  �  �  �  |  �  J    �  �  �  �  �  �  �  �  L  R  &  �  b  �  �  e  m;�o<49X;ě�;D��;o$�  �o���
�u�D���T����t��T������e`B�e`B��9X�u��o����C���t����
��9X�ě���j��j��9X�8Q�ě��ě����ͼ��ͼ��ͼ�������h��/����h��h�\)��h�����49X�o���o�C��+�\)�\)�t���P�'�w�<j�''49X�',1�,1�0 Ž49X�49X�@��@��H�9�aG��T���]/�ixս���q���y�#��t���O߽��P���-�����

���������wz�������zxnwwwwwwww��������������������JOW[_\e_[YWOJJJJJJJJjnrz����znjjjjjjjjjj�������


�����������������������������������������������
#

�����������������������������������������&)66BCDB6)$!&&&&&&&&
#%#!
�������������������������� ������������
#(-)#
inv{����������{unhei5;HLTmrpqnpmTKJC;335����������������������������������������#./05//#��������������������(,6IO[hrryy|yh[O6(&(pt������������ztpmlp��������������������_afntyz}znnmfa______fhit~����ytnhhffffff#/<HLPPHE</$#
������$ ��������_almvz�����zwma`____)5@BEB95)KNS[grtutohg[NHGKKKKWdt������������tg[WW����������������������������������������MN[got����xtrg[PNHMM����� 

������*/01;<HQU[ZYUMH</)**��������������������.5BINSYZNB8520......����������������������������������������ku���������������ukSTWamnz~zyvrmjaXTQQS|�����������||||||||9<IUabfhib`UI<<64399anz����������znia^^a������������������Y[htt���tmh[W[`[YYYY��5BQhrrhNI5�����������������������o{�����{rooooooooooou{�������{}{uuuuuuuu����������������agt�����ytgca]aaaaaa���������������	
#0<IMRLI<30#
	����

������������������������������� �����������������������������������������������������������������������������������������������������������������������������������������������������)*05=BJB5)$Yglt�������}ztgd[SOY
#000240#
��������������������KNQ[_c[VONMGKKKKKKKK�����������������������������	�����[[hisnhe[ZQS[[[[[[[[#0260(#!���������������������������������������������������������������$')(%�����#/H[UHE>64-#����ùõöùû�������������������������������������������������������������������H�G�B�C�H�U�a�j�n�s�z��z�n�a�U�H�H�H�H�;�1�/�,�/�;�H�T�U�T�H�A�;�;�;�;�;�;�;�;�U�P�H�F�F�H�U�U�W�V�U�U�U�U�U�U�U�U�U�U�M�F�4�/�/�&�(�4�A�L�Z�g�����w�o�f�Z�M�t�s�h�t¦©²¶¸²¦���������������������������������������������ĽͽӽܽܽнȽĽ��� ������������$�0�4�:�9�1�0�$���àÓÇ�y�n�l�n�}ÄÇÓàìùþ��üùìà�#�!�"�#�-�/�<�>�E�?�<�/�#�#�#�#�#�#�#�#�ݽԽѽٽݽ����������ݽݽݽݽݽݽݽݿ�����޿ӿοѿڿݿ�����������������%�(�5�A�N�N�N�L�H�A�5�(����ŭŪŬŭŴŹ������������Źůŭŭŭŭŭŭ����y�w�u�y�����������Ľҽ۽ҽĽ�������������&�5�N�g�s�������s�g�N�4�(���z�y�m�i�m�z���������z�z�z�z�z�z�z�z�z�z�ѿǿĿ��Ŀǿѿݿ�����	�������ݿ��A�>�5�/�(�(�(�5�A�A�G�G�A�A�A�A�A�A�A�A�*�(������*�/�6�7�6�*�*�*�*�*�*�*�*�"��	����������.�;�G�`�r�p�h�Q�;�.�"ŔŎōŔŝŠŭŹ��������������ſŹŭŠŔŔœŇ�{�x�v�{ŇŋŔŠťŭŭůŭťŠŔŔ���������������������������������������˾������������������ʾϾʾ����������������O�6�/�)�,�.�3�6�B�O�_�h�u�t�h�c�f�d�[�O�������������ʼ���#�"��������ʼ��"����	��	����"�-�/�5�3�/�"�"�"�"�ѿп˿Ͽѿѿݿ������ݿѿѿѿѿѿѿ��������������������Ŀſ˿ɿĿ����������	����������������������#�,�2�@�/� ��	��������y����������������������������������������	�������	�����������������������������������������������m�d�a�T�H�G�D�H�J�T�a�m�r�z�}���{�z�m�m�"����	��"�/�;�H�X�a�c�b�Z�T�H�;�/�"�����%�*�6�C�L�K�C�>�6�*�������x�w�l�j�_�]�_�l�w�x�����x�x�x�x�x�x�x�x�	����������	���"�*�#�"��	�	�	�	�	�	�����������������������������žž�������������ƾƳƳƱ�������������������������ƧƟƚƎƋƆƎƚƧƨƲƭƧƧƧƧƧƧƧƧ�������(�4�A�K�M�S�U�M�M�A�4�(�����������3�@�L�Y�[�L�@�0�'������������3�M�f��������������Y�@����àÖÛààéìóòù����ùòìæàààà������������������������'�5�7�)��������������������������������������'�!�&�'�4�@�@�F�@�4�'�'�'�'�'�'�'�'�'�'�5�-�(��(�5�A�N�R�N�A�?�5�5�5�5�5�5�5�5��Ʒ����������������������������������������������� � ���������������F=F<F1F1F1F<F=FJFVFVFbFVFPFJF=F=F=F=F=F=����������������������������������������D�D�D�D�D�D�EEEEEEEED�D�D�D�D�D��T�G�;�.�+�"�����"�.�;�D�T�a�k�i�`�TßÞÙÙàù����������������������ùìßā�v�t�h�g�h�t�yāčĚĝěĚĎčāāāā��������ĿĴĿ����������������������������������
��#�)�0�#��
�����������������U�J�R�U�b�j�n�{Łń�{�z�n�b�U�U�U�U�U�U�L�F�D�E�L�M�Y�e�r�s�~������x�r�e�Y�O�L�"�����"�,�.�/�;�G�H�M�G�;�.�"�"�"�"������������������������������������ā�x�āčĔĚĢĦĪĬĪĦğĚčāāāāĚđčĉĆčĎĚĞĳĹĿ��������ĿĳĦĚ������������������%�%��������������������������������������������ìãàÙàìóù������ùìììììììì�#��
�������
��#�/�;�H�U�Y�U�N�H�<�/�#���������}�y�l�a�l�y�����������������������������ùϹ۹ܹ޹ܹϹù����������������!����!�-�7�:�;�:�0�-�!�!�!�!�!�!�!�!�S�M�Q�S�_�l�x�������������x�l�_�S�S�S�S������(�,�5�A�Q�Z�]�Z�V�A�?�5�(���������������������������������������������������������!�.�:�G�H�G�?�:�.�!�E�E�E�E�E�EuEmEnEsE�E�E�E�E�E�E�E�E�E�E� : < 6 a } < X j d $ N 7 # + , q @ n j - i q - E % ? 3 8 G X L 4 K I J a  1 * K ~ > g Z Z  b j d R I ( O m ) 5 Z M b 0 C L I Y 5 A T e > w k x N N ' U x B v  �    )  n  �  9  Y  �  9  �  �  d  Q  p  d  �  �  �  �  �  *  l  b  6  �    �  R  p  �  I  �  �  �  w  k  z    8    2  �  r  �  d  w  �  M  p  �  �  �  �  M  \  L  �  j  j  �  �  �  �  5  �  �  ~  �  �  �  �  �  j  G  �  6  g  �  �  f  C  �  �  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  P  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  m  e  ]  U  M  E  �  �  {  X  2    �  �  �  �  �  �  �  r  Q    �  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    !  *  3  <  E  O  Y  e  p  {  �  �  �  �  �  �  �  �  �  2  .  *  "      	  �  �  �  �  �  �  �  �  �  l  U  B  .    �  �  �  �  �  }  _  G  ,  	  �  �  �  ^  7    �  k  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  o  b  T  F  7  b  p  y    �  �  �  �  �  �  �  Z  >  '    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  [  -  �  �  �  N  �     �  �  �  B    �    �  �  �  �  �  b    �  �  :  �  (  &  '  ,  0  9  D  P  \  a  d  d  ^  X  P  F  6  &      b  ]  X  S  M  G  @  6  +      �  �  �  �  n  H     �   �      &  +  -  ,  %      �  �  �  b  4  
  �  �  O  �  ~  z  t  n  g  `  Y  R  J  A  4  #    �  �  �  �  l  <     �  !  #  $    �  �  �  �  �  k  J  '    �  �  �  �  k  >    �  �  �  �  �  �  �  �  �  �  �  ~  K    �  q    �  $  �  t  n  h  ^  V  S  M  B  6  (      �  �  �  �  �  �  k  J                         �  �  �  �  �  �  �  �  �  +  `  �  �  �  �  �     �  �  �  �  \    �  k    n  �  ^  �  �  r  `  M  :  (      �  �  �  �  �    �  �  �  �  �  _  ^  ]  \  Z  V  J  >  2  &    �  �  �  �  �    c  F  *  Q  O  F  <  /         �  �  �  �  g  ,  �  �  @  �  F  4  Z  i  q  n  i  `  T  @  %    �  �  x  E    �  �  A  �   �  �    '  *  $    �  �  �  �  q  H    �  �  �  ]  %  �  �            
  �  �  �  �  �  �  y  T  (  �  �  �  [  $  !  7  K  U  Y  Y  N  @     �  �  �  {  L    �  �  �     �  �  �  �  �  �  �  �  k  V  8    �  �  �  G    9  \  �  �  i  �  �  �  �  �  �  �  �  �  �  [    �  %  z  �    e    �  �  �  �  �  �  �  �  �  �  �  �  �  |  m  _  P  B  3  %  p  h  _  S  G  E  E  @  :  /  !    �  �  �  �  m  =  �  �  y  y  x  t  k  `  Q  @  *    �  �  �  }  T  -  	  �  �  �  {  m  m  t  j  ^  O  >  0      �  �  �  �  Y    �  ^  \  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  \  :    �  �  �  �  v  
         &  )  (  "      �  �  �  �  p  N  (    �  �    I  a  i  d  P  7      �  �  �  n  ?    �  �  A  �  �  3  '      �  �  �  �  �  m  L  '  �  �  �  �  y  `  v  �  �  �  �  �  �  �  �  �  i  Y  I  3    �  �  ~  #  �  4  �          �  �  �  �  �  �  �  h  ,  �  �  �  F    �  �  �  �  �  �  �  �  �  �  i  P  2    �  �  �  p  ?    �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  7  �  �  k     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  Q  8      �  �  �  �  t  f  X  H  3    �  �  �  �  v  J    �  �  {  A    �  �  �  �  �  �  �  �  �  �  �  }  t  k  c  [  S  K  C  ;                    �  �  �  k    �  4  �  �    �  �  �  �  �  �  �  �  R    �  n    �  ~  T  �  �  �    �  
�  O  S  ;    
�  
y  
0  	�  	�  	�  	5  �  -  �  �    �  �  0  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �     �  �  �  �  x  q  `  ;    �  V  �  �  U    �  3  l  O    1  )         �  �  �  �  �  �  x  _  F  -    �  �  �    �  �  �  �  �  �  �  �  �  {  n  `  P  >  +       �   �   �  �  �  �  �  �  v  j  ^  R  G  6     
   �   �   �   �   �      g  I  ?  6  +  !      �  �  �  �  �  �  �  �  �  y  C     �  ;  0  &      �  �  �  �  �  �  l  W  ?  #    �  �  �  N  }  �  �  �  �  �  �  �  m  E  "  �  �  �  S    �  �  Z    O  E  ;  .    	  �  �  �  �  �  �  �  �    k  W  0     �  �  E  �  �  �  �  n  4  �  �  E  
�  
Z  	�  	(  S  J  �  �  �  �  r  _  K  6       �  �  �  �  c  =    �  �  S    �  �  �    {  q  f  Z  M  ?  0    
  �  �  �  ~  M    �  �  a  i  r  w  y  {  s  b  J  #  �  �  �  p  E    �  �  �  �  l  �  �  �  �  �  �  �  �  �  �  �  �  q  _  M  ;  '    �  �  J  <  /  #      �  �  �  �  �  �  �  ^  :    �  �  �  *          �  �  �  �  �  �  �  w  V  3    �  �  �  q  G  �  �  �  }  q  c  R  >  &  
  �  �  �  r  D    �  �  N  �  �  �  }  r  e  W  H  7  &    �  �  �  �  �  `  8     �   �  �  Y    �  �  �  �  j  @    �  �  �  `  #  �  �  X    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  U  @  *    �  �  �  �  �  z  i  P  4    �  �  �  v  R  0        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      %  4  C  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  Q  �  �  �  �  �  �  �  �  w  ^  D  +    �  �  �  T  %   �   �  L  4      �  �  �  �  [  *  �  �  �  �  L    �  d    �  R  J  B  :  +      �  �  �  �  �  �  �  �  �  �  v  h  [  �            �  �  �  �  %    �  �  �  �  n  :    ;  �  �  �    s  c  R  A  /      �  �  �  �  �  �  �  �  �  b  ^  Z  U  J  @  4  (        �  �  �  �  �  �  q  "  �  �  �  �  �  �  �  �  �  �  T  #  �  �  ~  I    �  `  �  R  �  }  r  h  ]  S  H  >  4  )        �  �  �  �  �  �  �  e  `  X  M  @  +  
  �  �  �  s  A  	  �  l  
  �  ^  g  j  m  H    �  �  p  K  #  �  �  �  �  g  �  �  �  $  �  T  