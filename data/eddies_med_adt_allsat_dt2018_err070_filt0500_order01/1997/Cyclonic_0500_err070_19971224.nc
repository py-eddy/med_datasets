CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��1&�y      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N5�   max       P�|f      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��   max       <�o      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?z�G�{   max       @Fu\(�     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�\    max       @v|          	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q@           �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @��           �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �bN   max       <o      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B2�%      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B2��      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?vPX   max       C���      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?q��   max       C��f      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          q      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       MҞs   max       P�T�      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��=�K^   max       ?�($xG      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��   max       <�o      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?}p��
>   max       @Fu\(�     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�\    max       @v{��Q�     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @Q@           �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @�S`          �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Dt   max         Dt      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?~Ov_ح�   max       ?�($xG     �  V�         p   Q         2         3                     !                              <   8         ,            3                     
                                    2                           N�ZNp�0P���P�|fOAN��2O�^�N�X5N�PU��O3�N�Np��NK�XO�VgN)KWP��N��DN]��N�jZNJM�O*�TN�[4O�+�N�]yN��APl�&O�[1N5�NV�CO��N�%�N�-�O��P&!CNH �N��O/B�N�nN��]N�24N�(�N���Nņ�OY��O��N�m�N�2�NI�NBr�O9hO-3�NQ�O�qvN�'�NG��O:#{N�u�N��N�A,N�xO7q�N�6(<�o<#�
;D��:�o��o�#�
�#�
�49X�e`B�e`B��o���㼼j�ě��ě��ě����ͼ�/��/��/��`B���o�+�C��C��C��C��C��\)�t����8Q�8Q�P�`�Y��Y��]/�]/�m�h�m�h�q���u�y�#�}�}󶽁%��%�����������\)���㽛�㽝�-���w�� Ž�Q�ě���������/��*/2<AHHOHG</%(******!)5BLB@5)!!!!!!!!!!�<UbjoqqhUI<������#<U�����bUI<0����)*0330--)#LOT[dhihd\[TOKIHLLLLot����������������{ooz������������ztoooo����������������������������������������
#/HUYUPH<1-#
HHIUaba]XURLHEHHHHHH[\hu����uh\V[[[[[[[[)+5?<51)KN[gt����������kdYNK��������������������T^amuz�������zmaWLOT��������������������&)/6<BGOQOB62)&&&&&&��������������������)/6=@61))5;BCCA==;750)% NUabaaabnz}�zneaUMNNrz��������������vqpr����������������������������������������EO�������������[NGDE#/<HU`hjjnvxaUH/##}��������}}}}}}}}}}fnz����znkffffffffff��������������������������������������������

�����������������������������#3BN[g�������g[T5$#//2<BEC<6/.-////////IO[ahkqt}{th[TPQTOII������������������������������������������������������������"#'-/8<<DFC><8/,%#""��������������������+6BBCOS[_ba[[ODB96++jn{��������|{ztngdij�������
 
��������������
������������������������������������������������wz��������zxwwwwwwww�����������~����������������������������qt�������������zxupq�����������������������������������������

 ����������������������=BNgt~������yg[NEB==}��������������z}}}}��! 
 ���������� ���������� ��������������������$)5BGNQNMB50)!$$$$$$�	���������	����"�-�.�"��	�	�	�	�	�	�Ŀ������Ŀοѿؿ׿ѿĿĿĿĿĿĿĿĿĿĽy�j�j�o�u�����Ľ���?�I�F�"���ݽĽ��y�������T�D�=�,�Z�������������	�� �������ɺúºɺɺֺ���������������ֺɼ������
����'�*�4�8�4�'������Y�R�K�:�7�A�M�f�r�������������������Y��������������������������������������H�A�<�8�2�2�<�H�U�_�a�f�f�a�`�a�^�U�H�H�T�>�5�3�6�G�T�y�����ĿϿͿڿ��ֿ��y�T�O�H�B�6�/�*�)�1�1�6�:�B�G�W�O�O�Q�S�S�O���������������������������������������ž����������������������������������������g�d�^�Z�T�Z�g�o�s�y�v�s�g�g�g�g�g�g�g�g�ݽڽ޽ڽڽ�ݽ�����&�,�)�����ݻ������������ûϻû�����������������������������������������������4�=�6�"�����ȿ"� ��"�)�.�;�D�G�N�G�A�;�.�"�"�"�"�"�"�m�e�`�X�`�e�m�u�y�����~�y�q�m�m�m�m�m�m�ʾȾ������������ʾվ׾����׾ʾʾʾʿ;�0�1�.�'�.�;�B�G�M�J�G�;�;�;�;�;�;�;�;���������������	��"�/�;�H�Q�H�A�/�"�	���#�����
�	�
����#�/�-�/�0�;�/�#�#������ĸĪĳ���
�#�.�0�<�?�0�+����������n�k�b�U�M�T�U�b�n�{ņŅ�{�q�n�n�n�n�n�nŔŊŔŚŠŭŹ����������������ŹŭŠŔŔ�v�g�N�D�M�g�t²������������²�@�3�.�0�6�A�M�Z�s�{��������������s�f�@�����������������������������������������H�C�>�?�H�U�Y�Z�V�U�H�H�H�H�H�H�H�H�H�H�a�Z�H�<�6�:�I�U�a�j�n�zÇËÏÍÇ�z�n�a�B�<�9�B�N�O�[�h�t��y�t�h�[�Z�O�B�B�B�B�����������������������������������������L�K�H�L�O�Y�a�e�r�z�~�������~�r�e�c�Y�L���ݾϾӾ۾���	�"�.�A�I�L�G�;��	����D�D�D�D|D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D���������������'�,�+�,�'����������C�:�6�-�*�)�0�/�6�C�Q�\�f�h�o�g�\�U�O�C�b�X�Z�b�n�z�{�{�{�n�b�b�b�b�b�b�b�b�b�b�t�r�h�b�[�[�Z�[�d�h�l�tāāććā�z�t�tFFE�E�E�E�E�E�E�E�FF$F+F1F7F1F$FFF��������(�5�A�D�D�A�;�5�/�(��������������������������������������������������������ûллܻ����ܻٻлû�����ݿѿĿ������¿ſѿݿ�������������A�5�(���(�:�D�N�Z�g�s���������s�g�N�A���������������������	�����	���������;�4�/�'�/�3�;�>�H�P�P�N�H�A�;�;�;�;�;�;������������������������������������������������� ��������������ƚƒƎƁ�~�vƁƁƎƐƚƧƳƻ������ƳƧƚƳƪƯƳƹ��������������������������Ƴ����������$�'�*�3�;�3�.�'��������L�C�J�Y�^�e�r�~�����������������~�e�Y�LE*E&EEEE'E*E7ECEFEPEREPECE7E5E*E*E*E*�Y�O�O�V�Y�f�n�r�t�r�h�f�Y�Y�Y�Y�Y�Y�Y�YĳĭħĢĠĦĭĳĿ������������������Ŀĳ�H�C�?�H�K�U�a�n�z��z�n�n�a�`�U�H�H�H�H�������������Ľн۽ݽ����ݽؽн��������S�O�G�?�:�9�6�:�;�G�S�]�`�j�a�`�S�S�S�S�������!�%�.�2�.�!����������������������'�,�0�+�'�����ÇÄÄÄÇÎÓÝàèãàÛÓÇÇÇÇÇÇ 3 P 1 P Q X 2 f R 2 i Y d \ Z v > - ^ m Y f f S B q A 1 � f ! Z V 6 @ b P ) > @ a C N H : C C N 5 b H ` � 5 C ( O X ^ [ h  N    �  �  �  H  S  �  �  �  3  ~  �  `  �  n  �  r  �  �  �  �  i  �  �  V  �  D    �  �  �      N  J    U    s  &    d  	  �  �  �  j  �  �  Y  >  6  �  �  �  �  X  �  �  U  �  -  y  �<o;��
��;d������`B��1�q������+��o���ě���/��/�t���h�m�h�+�C��C����0 Ž��u�#�
�'����Q��P�0 Ž��
�H�9�u�}����u��O߽�\)�u������ Ž�O߽�C���7L���w��Q콗�P��C���C���t�������{���T�%��񪽩���"ѽ�
=������l��V�bNB�nB,5B%�zB&��B��B �B��B� B ��B+oB7�B�[B2�%B��B
E)B-A���B��B�NB ϽBV�B�Bp�BR�B�6BmfB
��B�B%�BF\BެB6�BB!9�B	q�B��B�BJ�B�ZB6�B@B��B��B(��B��B��B~�BzxB 4�B �B �B
��B�BXrBУB?�B	B
�aB��B�B~WB�Ba*B��B��B%Q�B&��B��B?�B -�BQ5B ��B*��BؐB��B2��B4�B
�
B>�A��B�B��B ��B9�B46B��B ��B��B�B@B�B>kBrZB�LBDB�B!?�B	��B�CB��B>�B9qB�9B?�B�yB>B(��B�HB��B@B�\B <B ��B9�B
��B�fBAB��BAvB��B
�	BAyB��BG.B��BBAA\�AycA(�^A� �@Daj@ş�@݌NA�k(A�̍AmC�A�8!A��5ALiA�xiA1#@�r�A���Aa�GAk�<AP��Ab��A��~A��A�"A�D<A�SXA��A>�VAG8VA���A���A�hCA�?�u�A\FC��*?vPXB �dA�5�A�8OC���A�lA��H@���A}k*A�XA� A��)B��A��BZ�B�`?~�o@�C��]@��A⁪A�ɭA'�AL�Ay8@���A�a'A\�AyA$��A�y;@D% @� �@���A�zWAĀ�AlrjAװA���AL�A�A1`�@��zA��Aa�YAk !APzOAd� A���A�nA��A��kA�x�A��A>�JAI7�AĳA�yA�<A��T?�2�A[�4C���?��NB7�A�z�Aۀ�C��fA��A�~@�DA{��A��A��ZA��B��A�e<B��B{�?q��@TC��:@��fA�m�A�z�A'��AI�A|@�6JA��   	      q   Q         2   	      3                     "                              =   9      	   -            3                        	                                 3                                     =   A         #         /               !      )                     #         1                        )                                                                                             3   =                                       #                     #         -                                                                                                            N�ZNp�0PK��P�T�N��ZN��2O�N�X5NQ\�O��^O3�N�Np��NK�XO�� N)KWO�f�N6̯N-�N�jZNJM�O*�TN��DO�+�N�]yN��AP+��OR�MҞsN!mQOu��N�%�N}��O��O��NH �N�u$O/B�N�nN��]Nň�N�(�N���Nņ�OY��O�nNN�m�N�2�NI�NBr�O9hO-3�NQ�O��}N�'�NG��O:#{N}	vN��Nm�6N�xO7q�N�6(  B  �  �  �  �  �    	    Z  �    
  h  �  1  6  �  p  �  �  M  ,  /  7  �    �  �  �  	l  �     �  �    �  P  �  v  3      �  N  �  �  �  F  |  �    d  �    v  ,  �  �  k  �  �  �<�o<#�
��9X�t��t��#�
�\)�49X��1��w��o���㼼j�ě����ͼě���/��h��`B��/��`B���+�+�C��C��8Q�P�`�\)��P��w���L�ͽ8Q콑hs�Y��]/�]/�]/�m�h�y�#�q���u�y�#�}󶽅���%��%�����������\)���㽥�T���-���w�� Ž�j�ě���"ѽ����/��*/2<AHHOHG</%(******!)5BLB@5)!!!!!!!!!!�
#<UbgfbYYTI<#���#<b{�����`U<0
���)+..*)LOT[dhihd\[TOKIHLLLL��������������������oz������������ztoooo����������������������������������������
#/HUYUPH<1-#
HHIUaba]XURLHEHHHHHH[\hu����uh\V[[[[[[[[)+5?<51)R[gt���������tof[XR��������������������Tamz���������zmaXMPT��������������������')368BFOPOB64)''''''��������������������)/6=@61))5;BCCA==;750)% PUadnz|�zna`UOPPPPPPrz��������������vqpr����������������������������������������OTb������������g[NKO"$*/<HUX__ZURH</'#!"��������������������jnz����znmjjjjjjjjjj���������������������������������������������


������������������������������AN[gt��������tg[PE>A//2<BEC<6/.-////////Q[`hjpt|zth[UQRUQQQQ������������������������������������������������������������##(//<@CCA<<4/-&####��������������������+6BBCOS[_ba[[ODB96++jn{��������|{ztngdij�������
 
����������������
	�����������������������������������������������wz��������zxwwwwwwww�����������~����������������������������qt�������������zxupq�����������������������������������������

 ����������������������=BNgt~������yg[NEB==������������|��! 
 ������������������������ ��������������������$)5BGNQNMB50)!$$$$$$�	���������	����"�-�.�"��	�	�	�	�	�	�Ŀ������Ŀοѿؿ׿ѿĿĿĿĿĿĿĿĿĿĽ������y�w�}�����Ľݽ���(�7�4����ݽ��������s�Z�K�A�?�D�W����������������������ֺͺɺɺɺҺֺ��������������ֺֺֺּ������
����'�*�4�8�4�'������Y�Q�O�O�M�L�M�Y�f�r�y�������z�r�f�Y�Y��������������������������������������H�H�>�<�8�<�H�U�Z�[�W�U�H�H�H�H�H�H�H�H�y�m�`�T�J�D�F�G�T�`�m�y���������������y�O�H�B�6�/�*�)�1�1�6�:�B�G�W�O�O�Q�S�S�O���������������������������������������ž����������������������������������������g�d�^�Z�T�Z�g�o�s�y�v�s�g�g�g�g�g�g�g�g���ݽܽ��������%�+�(�������點�����������ûϻû���������������������������������������������"�2�:�4�"�����Ͽ.�)�"��"�.�;�G�J�G�<�;�.�.�.�.�.�.�.�.�m�h�`�\�`�g�m�w�y�~�~�|�y�o�m�m�m�m�m�m�ʾȾ������������ʾվ׾����׾ʾʾʾʿ;�0�1�.�'�.�;�B�G�M�J�G�;�;�;�;�;�;�;�;���������������	��"�/�;�H�Q�H�A�/�"�	���#�������#�+�+�.�/�9�/�#�#�#�#�#�#������ĸĪĳ���
�#�.�0�<�?�0�+����������n�k�b�U�M�T�U�b�n�{ņŅ�{�q�n�n�n�n�n�nŔŊŔŚŠŭŹ����������������ŹŭŠŔŔ¦�h�c�i�t����������������¿¦�Z�M�A�;�4�5�>�A�M�Z�f�n�s�����z�s�k�f�Z�����������������������������������������H�F�@�A�H�U�W�X�U�U�H�H�H�H�H�H�H�H�H�H�a�^�H�?�9�<�L�U�a�n�zÁÇÉÍÌÇ�z�n�a�B�<�9�B�N�O�[�h�t��y�t�h�[�Z�O�B�B�B�B�����������������������������������������L�K�H�L�O�Y�a�e�r�z�~�������~�r�e�c�Y�L����������	��"�.�3�6�8�5�+�"��	��D�D�D�D|D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�������������'�+�*�+�'������������C�:�6�-�*�)�0�/�6�C�Q�\�f�h�o�g�\�U�O�C�b�X�Z�b�n�z�{�{�{�n�b�b�b�b�b�b�b�b�b�b�t�r�h�b�[�[�Z�[�d�h�l�tāāććā�z�t�tFFE�E�E�E�E�E�FFF$F%F1F4F1F$FFFF��������(�5�A�D�D�A�;�5�/�(��������������������������������������������������������ûллܻ����ܻٻлû�����ݿѿĿ������¿ſѿݿ�������������N�A�5�(�!� �(�5�@�J�Z�g�s��������s�g�N���������������������	�����	���������;�4�/�'�/�3�;�>�H�P�P�N�H�A�;�;�;�;�;�;������������������������������������������������� ��������������ƚƒƎƁ�~�vƁƁƎƐƚƧƳƻ������ƳƧƚƳƪƯƳƹ��������������������������Ƴ����������$�'�*�3�;�3�.�'��������e�Y�N�O�Y�c�e�r�~�������������������~�eE*E&EEEE'E*E7ECEFEPEREPECE7E5E*E*E*E*�Y�O�O�V�Y�f�n�r�t�r�h�f�Y�Y�Y�Y�Y�Y�Y�YĳĭħĢĠĦĭĳĿ������������������Ŀĳ�H�F�@�H�M�U�a�n�o�n�l�a�]�U�H�H�H�H�H�H�������������Ľн۽ݽ����ݽؽн��������S�Q�G�@�:�:�:�G�S�\�`�i�`�]�S�S�S�S�S�S�������!�%�.�2�.�!����������������������'�,�0�+�'�����ÇÄÄÄÇÎÓÝàèãàÛÓÇÇÇÇÇÇ 3 P = L 0 X 7 f K 3 i Y d \ Q v 7 6 h m Y f r S B q E  � _   Z O 6 : b M ) > @ V C N H : @ C N 5 b H ` � - C ( O > ^ J h  N    �  �  y  �  �  �  E  �  r  =  �  `  �  n  ?  r  :  R  �  �  i  �  �  V  �  D  ,  �  \  k  �    �  J  N  U  �  s  &    �  	  �  �  �     �  �  Y  >  6  �  �  n  �  X  �  �  U    -  y  �  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  Dt  B  <  5  -  #        �  �  �  �  �  �  �  �  �  |  N  !  �  t  b  P  =  )      �  �  �  �  �  �  
      
    �  �    X  �  �  �  m  C    [  H  "  �  �  �  r  $  �    �  �  �  �  �  �  �  S  
  �  �  l  [  1  �  ~  �  o  �  Z  /  [  d  �  �  �  �  �  �  �  �  {  I    �  �  [  �  ~  �  q  �  �  �  �  �    x  q  b  Q  :      �  E    �  �  �  \  c  �  �  �  �  �  �  �  �        �  �  a    �  0  y  X  	  �  �  �    +        �  �  �  �  �  �  �  �  R  �  �    d  �  �  �         �  �  �  �  �  t  :  �  �  X    �  �    @  P  U  N  D  J  S  Z  V  I  1    �  �  Q  �     �  �  �  �  �  �  �  �  �  �  q  [  A  "  �  �  �  Y  *        s  g  [  M  6     	  �  �  �  �  �  �  �  l  S  :  !    
    �  �  �  �  �  �  �  �  �  �  �  �  |  t  m  e  ^  W  h  a  Y  R  K  D  <  0  !      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  b  L  '    �  �  �  �  e  C  !    $  1  !      �  �  �  �  �  y  i  Z  M  D  :  0  &          4  ,       �  �  �  o  L  $  �  �  r  <    �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  q  V  :      �  �  h  k  n  o  j  f  W  ?  (    �  �  �  �  W  ,     �   �   u  �  �  �  �  �  �  �  �  �  y  o  c  U  G  ;  1  '    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  M  0      �  �  �  �  �  �  w  k  _  U  <    �  �  &  �       &  +  (  #           �  �  �  �  �  h  >    �  �  /  -    �  �  �  j  6    �  �  I  �  �  �  h  &  �  �  c  7  0  (  !      �  �  �  �  �  �  �  �  �  �  ~  s  h  ]  �  �  �  �  �  �  �  �  �  r  _  L  <  -    �  �  �    �    �  �  �    �  �  t  !  �  �  q  L  %  �  t  �  P  �  y  �    )    �  �  �  �  z  P    �  �  $  �  .  _  X  ~  8  �  �  �  �  �  �  �  �  �      (  9  K  `  t  �  �  �  �  x  |  �  �  �  �  �  �  �  �  �  �  �  �    u  j  T  -    	]  	h  	j  	g  	Z  	A  	"  �  �  �  :  �      �  3  �  �  �  �  �  �  �  �  �  �  z  k  \  M  ;  !    �  �  �  �  �  O  �                       �  �  �  S     �  �  k  3  �  �  �  �  �  �  q  T  3    �  �  �  V  #  �  �  �  K    �  �  �  �  �  �  �  �  �  �  �  �  �  d    �  g    �  x  $        �  �  �  �  �  �  i  L  -     �  �  ?  �  �  v  B  �  �  �  �  �  �  �  �  �  �  �  �  L  �  {  �  e  �  B   �  P  <  )    �  �  �  �  �  �  w  T  .    �  �  e    �  K  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    E  v  s  f  S  A  .      �  �  �  �  o  L  &  �  �  �  �  �  �    2  &    �  �  ~  ?  �  �  3  �  .  �    Z  �  �   �    �  �  �  �  �  s  X  ;    �  �  �  �  d  5    �  �  �    	  �  �  �  �  �  �  �  y  e  P  ;  (        '  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    x  p  d  X  L  N  F  6     	  �  �  �  �  �  �  l  V  A  6  $    �  �  �  �  �  �  �  �  �  �  �  �  �  }  s  b  H  %  �  �  u  %  s  �  �  �  �  r  X  =  #    �  �  �  �  �  �  �  y  U    �  �  ~  x  q  k  d  ]  U  N  H  A  :  7  5  3  2  3  5  7  8  F  A  <  6  1  ,  &      	  �  �  �  �  �  L     �   �   J  |  g  S  >  &    �  �  �  �  �  �  v  V  3    �  �  q  1  �  �  �  w  g  T  >  &    �  �  �  �  s  H    �  O  �  P    �  �  �  �  �  k  ?    �  �  �  n  X    �  �  o  3  �  d  V  I  <  ,    �  �  �  �  s  M  5  '      �  �  �  Q  �  �  �  w  [  8  
  �  �  :  �  �  ;  �  �  "  �  �    �    
  
�  
�  
�  
`  
  	�  	F  �  `  �  r  �  y  �  |    �  6  v  l  c  Z  P  D  7  +    	  �  �  �  �  �  �  �  �  �  �  ,  	  �  �  �  �  |  ?  �  �  K  �  �  6  �  �  i  \  D  �  I  l  �  �  �  g  L  -    �  �  u  4  �  �  .  �  2  �   �  �  �  �  �  �  �  �  r  g  a  Z  T  B  )    �  �  �  k  Q  R  a  g  U  ;    �  �  �  �  l  M  /        1  M  ?     �  �  �  �  �  �  z  f  S  :  !  	  �  �  �      !      �  �  �  �  {  m  n  e  J    �  �  ]    �  p  �  0  X  Y  �  �  �  q  L  (    �  �  w  =    �  x  .  �    5  �  �