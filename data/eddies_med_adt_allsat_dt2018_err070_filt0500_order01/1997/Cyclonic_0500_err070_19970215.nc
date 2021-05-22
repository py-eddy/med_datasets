CDF       
      obs    L   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�n��O�<     0  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N 6   max       P��,     0  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =e`B     0      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?^�Q�   max       @F�33333     �  !<   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��G�z�    max       @v{�
=p�     �  -   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P            �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�Ӏ         0  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��F   max       ='�     0  :�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��0   max       B4��     0  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B4�\     0  =$   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =���   max       C��N     0  >T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =�3   max       C��4     0  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          H     0  @�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =     0  A�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =     0  C   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N 6   max       P�I     0  DD   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�l�!-w   max       ?ո���*     0  Et   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       =aG�     0  F�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?u\(�   max       @F�z�G�     �  G�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��G�z�    max       @v{�
=p�     �  S�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P            �  _�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��`         0  `,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C<   max         C<     0  a\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��O�;dZ   max       ?շX�e     �  b�         
   %         
      (   (   6               .   ;   H   	   *            
            "      	         $      5   7                  '      0                   &      
            
   
            1         
                                    N�1�P�UNĦO�n�N!�O��N���Nj��O�PD�O|�O'��O�U�N��Om7�P'FP�VP��,OޕPn��NDu�O���N�+KN��tN�K�N@��N�`5O�O��9O��N�4N;�%O�>�O�,�P�JP�|�N��3N��Ok��N���O���O�ZN���O���N�[�O��O�tJN�.O���P,��No��O+}�Nҽ�NE)8Ov,�ONO%��O9�UN�^�NN*�O).�N��O�[�N��*OM�\O�CENZJ�N�g�NyN��Nek�N 6Of�N�p�OV0N;��=e`B<o;ě�;��
;�o���
�ě���`B��`B�D���D���T���T���T���e`B�e`B�e`B�e`B�e`B�u��o��o��C���t���t����㼛�㼣�
���
���
���
��1��1��9X��j�ě�����������/��`B��h��h�������o��P�����#�
�#�
�'''49X�<j�@��@��@��P�`�e`B�m�h�q���q����o��7L��C���\)���㽛�㽣�
���
���T��E���E������������������������������
#)$�����������������������������'HU[`enz~znfU</(!'ONB646:BOUTOOOOOOOOO:<IOUbnrwyondbUI><9:�������������������������������;BN[t~���tg[NI?976;Ha���������mTHAFELLH��
%/452120/#���������������������}~����������������������������������������� &....*)���� IN[t�����������wgMGI����� 
�������������#&!��������$*-6CJMMKCA6*em���������������zme���������������������������������������� &)/.)��������������������������������������������
�������������BBFLORZ[b_[[OEBBBBBB
<UX[eii`UH/#

��)3.%#�������>BOT[_hijih`[OB>>>>>IOR[hpnjjkh[POFIIIII66ABNOTOMB6466666666q�����������������tqP[t������������tgYQP����
#71'
�������������������������������������������������uu������uruuuuuuuuuu����
���������#/7<?></)#��)+,,-*)������&)/57BN[afeb[N5+)&%&TTV]aimwvmia]TTTTTTT�����������������}��������������������invz{��������|zpnjii����������������������������������������!&.IUbn{���{bI<0!#)6Obtt����}OB;61)##������

�������<BDN[^gmt{tg\[NB>6<))-./-)��������������������ru�������������|utmr��������������������������������������������������������������������������������NN[gkjg[NJNNNNNNNNNN�����

����������GHUaihaULHGGGGGGGGGGJP[gty|{z{~�t`[NHFGJ����������������������������������������CUbn{����yrmbUIB>=?C###'+0000)#!####������������[[\gmkhgd[YY[[[[[[[[egklpt���������}tgee����������������������������������������������������������� #/7<>CD<7/(#!      DHUaoz~���}znaURMIGD����������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��f�A�&���$�?�X�f�s�����������������s�f���������������������ǾʾӾоʾȾ��������5�����A�\�s�������������z�s�g�Z�N�5�����������������������������������������������������������ʼּ׼����ּԼʼ��'�%������������������'�+�*�'ŔŉőŔŞŠŭŸůŭťŠŔŔŔŔŔŔŔŔ�����z�z������������ÿȿ˿ȿĿ������������������A�N�������������������Z�A�E*EEE"E*E7ECEPE\EiEuE�E�EuEiE\EPECE7E*Ç�z�n�f�a�W�a�n�zÇÝìùú��ùìàÓÇ������������6�C�O�\�h�k�g�\�O�6�*�����#���
���#�/�<�>�H�K�O�H�<�/�#�#�#�#����������(�5�A�M�N�Z�\�N�A�5�(�����޾ݾ����	�"�.�H�W�^�`�M�;�&�	����������������������������������������������f�\�s���ʼ���!�.�9�?�=�1�$��ּ����"���	���������	��"�,�.�4�5�4�.�"�"��������ƳƚƄƆ�ƙƯƾ�����$�-�-�������� �&�(�5�=�9�5�.�(���������Ŀ����������������Ŀѿؿ���������ѿ��$�!���$�0�8�=�I�M�O�O�I�F�=�0�$�$�$�$�)�#�)�-�3�6�7�B�K�M�O�P�O�B�B�6�)�)�)�)�U�S�H�C�C�H�U�U�a�b�n�r�t�v�r�n�a�W�U�U�����������������������������������������������޾�����	���!���	������������ɾ��ʾ���	��"�,�(�&�&�)���	�����_�S�R�U�[�e�k�n�zÇÓÚÝáàÓÇ�z�n�_�`�V�\�`�e�m�w�y�������������������y�m�`�Ϲǹù����ùϹܹ����������ܹϹϹϹϻF�F�:�9�:�:�F�O�S�_�\�S�F�F�F�F�F�F�F�F�r�k�e�Y�Q�L�C�B�L�e������������������r����µ°¯·������������������������ؿ��u�q�s�����������ݿ��������ݿĿ������~�Y�H�M�X�����ֺ����-�A�Q�C���ɺ��~�ݽڽнĽ������������Ľнؽݽ���߽ݽݿG�G�>�G�I�T�^�W�[�T�G�G�G�G�G�G�G�G�G�G���׾Ͼ˾׾���	��(�9�;�G�;�.��	�����6�,�-�1�6�:�B�O�W�T�O�L�B�8�6�6�6�6�6�6�[�I�@�?�<�C�O�[�h�tāĂćčĖčā�t�h�[��������������������)�1�5�6�:�)���;�8�/�"� �"�%�/�;�F�H�I�H�=�;�;�;�;�;�;�/�"�����"�/�;�T�a�i�n�n�h�a�]�H�;�/�{�o�p�o�n�o�{ǈǔǕǝǕǔǈ�{�{�{�{�{�{��x�s�l�s�����������������������������x�o�`�S�J�F�A�>�F�_�l�|���������������x�M�D�@�6�4�1�4�@�M�Y�[�f�j�f�f�Y�M�M�M�M�������j�a�\�Z�_�g�s�����������������������������������������	��-�*�����������������������������������ÓÉÇÁÄÇÎÓßìùþþùøöïìàÓ�Z�N�M�Z�Z�f�s�������������s�f�Z�Z�Z�Z�A�?�=�?�A�B�M�Z�^�f�Z�M�A�A�A�A�A�A�A�A��ŭŠşŞţůŹ�������������������������*�*�����*�6�C�O�O�\�c�d�\�O�C�6�*�*�����������������������������������������t�o�h�Y�N�O�T�[�h�tčĚĦĭİĦĚčā�t���������Ƽʼּ��������ּʼ������������������
����
��������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��H�F�E�D�H�T�W�Z�T�L�H�H�H�H�H�H�H�H�H�H���������������5�B�\�e�N�J�9�)����~�t�n�t¦²³¹µ²¦�
���������������#�:�<�J�?�<�0�#���
�û��������������м��'�-�&����ܻлýݽ۽ݽ���������������ݽݽݽݽ����������������������������#�0�<�<�<�;�0�#���������
��������������������������	�
���
�
�!�� �!�,�.�:�C�G�J�S�T�S�G�=�:�.�&�!�!�S�P�S�Z�`�l�y���y�l�d�`�S�S�S�S�S�S�S�S�������������������ûͻлջڻӻлû�����FE�FFFFF$F1F6F1F-F$FFFFFFFF�����������������ùϹܹ޹߹޹ԹϹù������@�9�<�?�@�M�S�Y�\�Y�O�M�@�@�@�@�@�@�@�@ Z D 3 H C = d b  Z L p P E 7 F 6 L < 9 r P / ` H F H 9 3 5 > B _ 5 % \ 9 Q i \ ' 6  3 ( H j B R K P N C X  V C O ; 3 P f Y Q 9 G o E P O ~ � & [ 6 >  �  �  �  [  L  W    �  P  �  <  �  �    �    �  �  /  �  �  I    �    \  �  -    A    M  �  z  �  �  3  ;  4  �      �  �  �  B  �  �  �  <  �  {  �  c  �  6  w  �  �  `  �  T  [  +  �  '  �  �  R  �  �  a  )  �  �  e='t��D�����:�o��1��o�T���<j�P�`��+�t��0 ż�����`B�q����t���{��j�e`B���
�o�t���`B�C��ě���9X�Y��'�h�t���h�ixս49X����������P��`B�P�`�C��8Q콋C��\)���-�0 Ž0 Ž�O߽T���e`B���-�<j�P�`�]/�D����%�aG��e`B�������ixս�
=�u��t���C�������E���������������� Ž�\��G���F����B��Bd�B4��BC�BbIB'2(B�gB�TB��A��B3B �@B�~B=�B��B
��B[�B,��B0%�B�/B�B*��B��Bc@B!�IBsBݕB{�B�MB�IBU�BM7B!G9BēB�@B�`B ��B3?�B�6B "B��B�~A��0BSkBA�B�B �aB"��B'rB�wB#��B�TB��B"�B
��B2NB7�B�^B�EB�B�PB֜B	MhBEGB�B(�B%<�B��B�%B	�Bl"B�9BN�BLiBkB2�B��B�\B4�\B(�BH}B'GBB�B��A�~9B?JB �|Bd4B?�BǡB
��B�B-|B0A�B>�B�nB*}B�9BA�B!�+B!B�Bk�BB�B=5B@&B?MB"�B�B�wB��B!#(B3F�B�B8%B�B HA���Bf�B@�BC�B ��B"�B&��BF:B#�GBRGB��B"|\B
��BG�B�~B��B�VBN�B��BƙB�}B?EB5�B'�B%6eB��BրB	�VB�+B"B@B?5B@BE[C�-/A?��AM��A���A�D�@�*?b^�A��As$�A�,C��A� �A��A�rNA��A\�}A���@�I'A]5�B��A�dAz�aB
x�AיtA�M�AI�-AY�AY��AȊ�Am��>�q@��\@ A�ՒAv�h@2�A(A{Af�AZ60A�GA�vhA��A� �A��^B�?AHD�@��4@��lA�տA�M@Y�iA˸�ABq2A=B�A��B ��A��A�ׯA �~A�*�C�A��A��A��DA���@�NA/�[A/��A��A��?A��A �@��C��N=���@�#vC�(�A@��ANJ�A�L�A�]�@��;?o�ZA�1As0eA���C���Aɐ�A���A�q�A��hA\��A���A��A[8/BK�A���Az��B
~A�~	A�^AI�AW	AY�A�p�An�>�)k@��?�$A���Ax�@@g;A(O�Ae�uA[o[A�u�Aڎ�Aӈ�A��A�{cBƞAH�@��@�
A�t�A���@Uj�A�t'AB��A=ܖA��WB �5A��A��K@�0�A�~3C�{A�\�A��OA��cA�hx@��)A0H�A0V�A���A��A��A;�@�,�C��4=�3@��         
   %               (   (   7               .   ;   H   
   *                        "      
      	   %      5   8                  (      0         !         &                  
   
            2         
                                          -      '                  1                  )   '   =      7      !                  !               '      )   =                        !         !      )   /                                                %                                    '                        '                        =                                             '      !   =                                       )   /                                                %                              N��iO��N�SOl&N!�N��N���Nj��OWP�FNЍO'��O&EN��OP.�O���O��QP�IOޕOwǳNDu�O"�
N���N��tN�K�N@��N�`5O�k�Oe�O��N�4N;�%O�>�O�,�O�PP�|�N���N��O%I�N���O���O6�NN���O| �NL�_N��O]yJNy�XO���P,��No��O+}�Nҽ�NE)8O`�,ONO%��N�f�N�q_NN*�O��N��O�[�N8��OM�\O�CENZJ�N�g�NyN��Nek�N 6Of�N�p�OV0N;��  �  �    `  �  �    *    1  	�  �  s  �  �  �  �  �     @  :  3  *  x      �  =  �  &  d  y  �  _    a  N  ;  o  {  �  �  	  *  o  .    L  �  �  �  �    �  �  0  k  �  �  ?  B  �  �  �  v  �  B  �  �  �  �  �  T  �  |  =aG�;ě�;�o�o;�o�o�ě���`B�ě�����\)�T�������T���u�\)���u�e`B�t���o��1���㼓t���t����㼛��ě���9X���
���
��1��1��9X�t��ě���`B�����o��`B��h�����0 Ž+�+�,1�#�
���#�
�#�
�'''8Q�<j�@��Y��H�9�P�`�m�h�m�h�q���}󶽃o��7L��C���\)���㽛�㽣�
���
���T��E���E����������������������������� % ������������������������������/5<@HU[bga^UH<4/+((/ONB646:BOUTOOOOOOOOO;<=IUbnnibXUTIB<;;;;�������������������������������@BMN[\gstttrjg[NEB@@QVaz�������zaTNJLPQQ��
 #'*##
��������������������}~�����������������������������������������$-,-,(����cgpt������������tgac����
����������������#%!��������$*-6CJMMKCA6*������������������������������������������������������������$)))&	 �������������������������������������������
�������������BBFLORZ[b_[[OEBBBBBB/<HTT`db[UH<#
���+,"�������>BOT[_hijih`[OB>>>>>IOR[hpnjjkh[POFIIIII66ABNOTOMB6466666666q�����������������tqP[t������������tgYQP�����

������������������������������������������������uu������uruuuuuuuuuu�����		�������#/7<?></)#��)+,,-*)������35?BNV[_a`][RNB5,+-3TTV]aimwvmia]TTTTTTT����������������������������������������jnzz����������zrnkjj����������������������������������������!&.IUbn{���{bI<0!#)6Obtt����}OB;61)##������

�������<BDN[^gmt{tg\[NB>6<))-./-)��������������������suw�������������~wts��������������������������������������������������������������������������������NN[gkjg[NJNNNNNNNNNN�����

����������GHUaihaULHGGGGGGGGGGJP[gty|{z{~�t`[NHFGJ����������������������������������������CUbn{����yrmbUIB>=?C###'+0000)#!####������������[[\gmkhgd[YY[[[[[[[[egklpt���������}tgee����������������������������������������������������������� #/7<>CD<7/(#!      DHUaoz~���}znaURMIGD����������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��A�4�$�$�0�N�Z�f�s�~���������������s�f�A�������������������ʾо˾ʾ��������������>�5�,�(��$�5�N�g�r�s�u�x�x�s�n�g�Z�N�>�����������������������������������������ʼ������������ʼּ߼���޼ּͼʼʼʼʺ'�%������������������'�+�*�'ŔŉőŔŞŠŭŸůŭťŠŔŔŔŔŔŔŔŔ�����������������������������������������(������(�5�N���������������s�Z�A�(E7E2E*E(E*E-E7E<ECEPE\E]E^E\EXEPEFECE7E7Ç�z�n�f�a�W�a�n�zÇÝìùú��ùìàÓÇ���������������*�6�>�C�E�C�B�6���#���
���#�/�<�>�H�K�O�H�<�/�#�#�#�#����������(�5�A�L�N�W�N�K�A�5�(������������� �	��"�$�.�<�@�=�2�%��	����������������������������������������������f�]�t���ʼ���!�.�8�?�<�0�"��ּ����"���	���������	��"�,�.�4�5�4�.�"�"������������ƾ��������������� �!�������� �&�(�5�=�9�5�.�(���������ϿĿ������������Ŀѿѿݿ�����ݿҿ��$�#��!�$�0�<�=�>�I�M�M�I�D�=�0�$�$�$�$�)�#�)�-�3�6�7�B�K�M�O�P�O�B�B�6�)�)�)�)�U�S�H�C�C�H�U�U�a�b�n�r�t�v�r�n�a�W�U�U�����������������������������������������������޾�����	���!���	�������������վϾ;׾���	��"�(�&�&�#�%��	���`�U�T�S�U�^�g�n�zÇÓØÜÞÝÓÇ�z�n�`�`�V�\�`�e�m�w�y�������������������y�m�`�Ϲǹù����ùϹܹ����������ܹϹϹϹϻF�F�:�9�:�:�F�O�S�_�\�S�F�F�F�F�F�F�F�F�r�k�e�Y�Q�L�C�B�L�e������������������r����µ°¯·������������������������ؿ����������������Ŀݿ����ݿѿĿ����~�Y�H�M�X�����ֺ����-�A�Q�C���ɺ��~�Ľ������������Ľнӽݽ��ݽݽнĽĽĽĿG�G�>�G�I�T�^�W�[�T�G�G�G�G�G�G�G�G�G�G�������׾Ӿ׾����	��!�1�1�.�"��	���6�,�-�1�6�:�B�O�W�T�O�L�B�8�6�6�6�6�6�6�[�I�@�?�<�C�O�[�h�tāĂćčĖčā�t�h�[��������������������!�)�,�,�)������;�8�/�"� �"�%�/�;�F�H�I�H�=�;�;�;�;�;�;�/�"���"�%�/�;�H�T�a�b�g�f�a�T�Q�H�;�/�{�y�t�w�{ǈǎǔǛǔǋǈ�{�{�{�{�{�{�{�{��y�s�s�����������������������������������x�t�e�_�R�K�P�S�_�l�x���������������M�H�@�8�:�@�M�P�Y�f�g�f�c�Y�M�M�M�M�M�M�������j�a�\�Z�_�g�s�����������������������������������������	��-�*�����������������������������������ÓÉÇÁÄÇÎÓßìùþþùøöïìàÓ�Z�N�M�Z�Z�f�s�������������s�f�Z�Z�Z�Z�A�?�=�?�A�B�M�Z�^�f�Z�M�A�A�A�A�A�A�A�A��ŹŭŢşťŭűŹ�����������������������*�*�����*�6�C�O�O�\�c�d�\�O�C�6�*�*�����������������������������������������h�^�[�T�[�]�h�t�āčĚĜĚėčā�t�h�h���������ȼʼּ߼������ּʼ������������������
����
��������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��H�F�E�D�H�T�W�Z�T�L�H�H�H�H�H�H�H�H�H�H���������������5�B�\�e�N�J�9�)���¢¦©²µ²¦�
���������������#�:�<�J�?�<�0�#���
�û��������������м��'�-�&����ܻлýݽ۽ݽ���������������ݽݽݽݽ����������������������������#�0�<�<�<�;�0�#���������
��������������������������	�
���
�
�!�� �!�,�.�:�C�G�J�S�T�S�G�=�:�.�&�!�!�S�P�S�Z�`�l�y���y�l�d�`�S�S�S�S�S�S�S�S�������������������ûͻлջڻӻлû�����FE�FFFFF$F1F6F1F-F$FFFFFFFF�����������������ùϹܹ޹߹޹ԹϹù������@�9�<�?�@�M�S�Y�\�Y�O�M�@�@�@�@�@�@�@�@ V > * ' C 6 d b   \ 8 p > E 4 B  K < ] r F # ` H F H 3 + 5 > B _ 5  \  Q c \ '    & H j = R K P N C X  V C > 5 3 Q f Y > 9 G o E P O ~ � & [ 6 >  �  f  �  �  L  �    �  -  �  �  �  �    �  ?  ;  �  /  !  �  v  �  �    \  �  �  �  A    M  �  z  �  �  �  ;  �  �    y  �  �  \    K  x  �  <  �  {  �  c  �  6  w    �  `  v  T  [  H  �  '  �  �  R  �  �  a  )  �  �  e  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  C<  �  �  �  �  �  �  �  �  �  �  �  j  P  4    �  4  Z  X  U  �  �  �  �  �  �  �  �  �  w  h  U  H  :    �  �  s  K  G                  	  �  �  �  �  �  �  y  ]  ?     �  �  �  �    :  M  Z  _  Y  E  "  �  �  l    �  M  �  z  s  �  �  �  �  �  �  �  �  �  �  �  �  �    v  l  c  Z  P  G  V  {  �  �  �  �  �  �  �  �  �  r  ]  D  #  �  �  �  M  �        �  �  �  �  �  �  �  �     �  �  �  �  �  �  �  e  *      �  �  �  �  �  �  �  �  �  {  \  ;    �  �  �  �  [  �  �    .  M  g  w    s  U  #  �  �  X    �  �  �  "  �    %  .  /  *  $       �  �  �  g  /  �  �  i    �  �  �  r  �  	  	E  	n  	�  	�  	�  	�  	_  	  �  j  �  j  �    E    �  �  �  �  o  V  9    �  �  �  n  :    �  �  F  �  �    �  �      &  ;  S  j  s  k  V  2  �  �  m    �  ]  �  C  �  w  N  $  �  �  �  �  z  ]  D  *    �  �  �  �  �  Z  -  �  �  �  �  �  �  �  �  �  �  �  �  g  M  4      �  �  Q  �  �  0  `  �  �  �  �  �  �  �  �  Y    �  a  �  ;  �  /  W  �  �  �  �  �  �  �  �  �  �  �  �  V  �  n  �  d  �  �  �  �  w  F     �  �  �  d  I  I  \  I    �  g  �  C  V  C                 �  �  �  �  �  �  �  �  `  <     �   �  �  �  �  �  �  �  �  �    2  =    �  �  _  w  9  �  �   �  :  1  '      
  �  �  �  �  �  �  �  �  �  {  l  ]  N  ?  �  �      )  0  2  +      �  �  �  �  ]  !  �  �  o  6  �    *  (         �  �  �  |  B  �  �  q  $  �  �  +  �  x  h  Z  ]  X  3    �  �  �  �  �  |  q  a  P  ?  .        �  �  �  �  �  �  t  T  7      �  �  �  �  �  �  m                  
    �  �  �  �  �  �  �  �  w  a  K  �  �  �  �  �  �  ~  r  c  U  F  8  )               �    ,  ;  :  3  *  "    �  �  �  �  h  3  �  �  [  �  \  �  �  �  �  �  �  �  �  w  Q  ,    �  �  N  �  s  �  &   �   s  &  !          �  �  �  �  �  �  �  �  �  ~  p  b  H  /  d  Z  N  A  1    �  �  �  e    �  �  �  �  4  �  �  ?   �  y  f  S  A  /      �  �  �  �  �  �  m  @    �  �  m  5  �  �  �  �  �  {  m  t  i  Z  I  7    �  �  �  8  �  l   �  _  =  %    �  �  �  �  �  �  u  R  '  �  �  v    �  !   �  �  �  �        	  �  �  �  �  [  !  �  �  a  $  �  7  �  a  O     �  �  �  \    �  U  �  �  C  �  �  �  j    �    ;  D  L  M  I  ?  .      �  �  �  �  f  ;    �  �  �  X  ;  '    �  �  �  �  �  �  �  q  a  Q  A  1  !       �   �  U  ]  j  n  o  l  e  \  O  ;  #    �  �  N  �  s  �  j  �  {  k  Z  J  ;  ,      �  �  �  �  �  �  p  K  &    �  �  �  �  l  N  =  6  5  5  1  )        �  �  �  �  w  �  �  �  p  �  �  �  �  �  �  h  A    �  �  B  �  �    �    �  	  �  �  �  �  �  �  �  �  k  S  <  "    �  �  �  �  }  a  �  �    %  )  *  )       �  �  {  B  
  �  w    �  �    N  Z  d  l  n  k  d  R  9    �  �  �  G    �  a  �  �  +  ,  .  -  (       
  �  �  �  �  S  '    �  �  �  �  �  �  �  �      �  �  �  �  �  n  2  �  �  ~  �  @  �  L  :  .  E  I  K  J  E  <  1  "    �  �  �  x  J    �  �  J    �  �  �  �  �  �  }  d  K  4      �  �  �  �  �  e  3   �   v  �  �  ~  i  X  T  K  >  .      �  �  �  I    �  d  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    s  h  \  �  �  �  �  �  |  d  J  /    �  �  �  ~  W  1            �  �  �  �  �  �  �  ~  ^  ;    �  �  |  A    �  �  t  �  �  �  �  �  �  �  �  u  f  U  E  4  "    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  I  $  �  �  _    y     �  0  !      �  �  �  �  �  �  �  �  �  j  J  %  �  �  �  N  k  g  d  a  _  W  M  ?  /      �  �  �  �  �  �  �  �  :  H  p  �  �  �  �  �  �  x  M    �  �  C  �  �  (  �  K  �  �  �  �  �  �  �  �  �  �  q  <    �  �  n  /  �  �  i  #  ?  6  -  $        �  �  �  �  �    '  a  �  �    N  �  =  A  /    �  �  �  s  9  �  �  ,  
�  
  	<  9  �  �  L  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  M  .     �   �   �  �  �  |  h  T  =  ,  C  ;  "    �  �  �  `  4    �  �  �  �  ~  q  f  ]  p  �  �  z  c  F  &    �  �  y  B  
   �   �  v  i  [  L  9       �  �  �  _  )  �  �  �  q  G  �  �  t  �  �  �  �  �  �  �  m  M  -  	  �  �  v  D    �  |     �  B  1       �  �  �  �  �  `  >    �  �  �  �  p  M  (    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  f  U  E  �  �  �  �  ~  {  w  u  t  r  q  o  n  w  �  �  �  �    2  �  �  �  �  �  �  �  �    j  \  T  L  C  7  ,  %  4  B  Q  �  �  �  x  g  U  C    �  �  ^    �  �  �  j  K  /    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  T  P  I  >  1    �  �  �  �  }  X  1    �  �  C  �  f   �  �  �  �  �  �  �  {  a  E    �  g  �  Y  �  5  �  �  K  �  |  n  a  R  6    �  �  s  6  �  �  l  #  �  G  �    {  �      �  �  �  �  �  �  �  �  m  H    �  �  m  1  �  �  {