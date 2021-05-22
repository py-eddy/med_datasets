CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�bM���        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N)   max       P��        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �Ƨ�   max       <D��        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?u\(�   max       @E�z�G�     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ۅ�Q�    max       @vg�z�H     
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P`           �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�P�            5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <o        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B-\        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B-^{        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�nZ   max       C��`        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��    max       C��[        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          A        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N)   max       P��        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?����l�D   max       ?���#��x        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �Ƨ�   max       <D��        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?z�G�{   max       @E�=p��
     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @vg��Q�     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P`           �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�P�            U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�qu�!�S   max       ?���#��x        W�            3      +               @            ;                     5   =      
   	   %            '   )                        @   	                                 	   
                  %            /   N��N!�PN�O�pyOT��O��N�`�NBkN�xFN�'�P��N)N�O#�(O�6O���O�O��kNdKO���NF�P���PN'LOGIO��O@pOyÊO���N��ZO�Oz�mO��O a@O���O^2N�JOF�O�8�N��P"��Np�FO3"�O+�3Ok��N���O^R�O�=	O���Og]�OŊOhwFN��O?O���Nr;OPZ�O�W�O=t�O�8�O�N*��N���O��N!S<D��<49X<49X;�o;D��:�o�D����o��`B��`B��`B�o�t��D���T���e`B��o���㼬1��9X��9X��9X�ě����ͼ��ͼ��ͼ�����`B��h��h���o�+�C��C��C��t��t��t��t����#�
�49X�49X�<j�<j�<j�<j�<j�D���L�ͽ]/�]/�aG��ixսq���q����o��7L��C���C���t�����Ƨ�))16BOTOEB:65,)('())���������������������
###
�����������)5BMPRVVXNB5)@BN[bglljig_[NB968;@����������������������������������������GHUabccaUQHFGGGGGGGGhhit����������thhhhh����������������������
#bn{�����{U0
�����

������������#/<GFDHKJHB</*&# ��������������������Welwz������zmaTPNQW�����).4)�������������������������������������������:<HSU\XUH<;6::::::::����������������������� ��������������������
O[ZI0
�����8<Haz��������xaUOD;8�������������JTamqxz���zmcaWTPKJJ����������������������������������������5<IO[ht���{ytWOB6475T[hntwxth[UOTTTTTTTTqy��������������tokq�����

������NW[ht����������[SKJN��������������������������������������������������������������������������������STVamz����zysmaUTRSV[gt���������tngYRPV�������������������������"!�������������������������������������������KOU[hmtplmnh[XOGGIHK0<IRUYbehfbUI@944500�������������������������������������Wgt������������rgZWWvz���������������|wv��������������������fgq�������������xtgfanz�����ynaUMH?CHLUawz�����������zrvwwww���  ������#/<HU\`]YUMHE-��������������������#05BHHD@<0&#"SUe{���������wnbULKS��� �����������������������������������������������������������������LN[goskg\[YNFDLLLLLL)16@MSROB6)
:<HMSNH<;9::::::::::�-�+�!����!�#�-�/�:�F�F�S�\�S�F�:�-�-�Z�U�Z�c�f�s�w���������s�f�Z�Z�Z�Z�Z�Z���������!������������N�=�5�1�4�A�N�Z�g�s�������������s�g�Z�N�������{�����������ĿʿѿԿӿɿĿ��������
� ��� �
��/�<�U�a�nÂÌÇ�z�n�a�H��
ŹŷŭŠŗŔœŔřŠŭŹź����������ŹŹ���������������������������������������ؼ�����|�z�t�y���������������������������t�~������������ľ������������������������w�e�G�E�O�h������������������������żŹŸŹ�������������������������������������������������������������������������������������'�4�:�:�4�-�'�������������������)�B�[�g�x�{�l�[�O�6����������������������������"�,�#��àÙÕÓËËÍÓàãìùü������ýùìà�����������|�����������������������������#� ��#�$�/�<�B�>�<�<�/�#�#�#�#�#�#�#�#�;�7�"� ��� �/�;�H�T�_�`�_�Y�T�O�J�H�;à×Þàìùýùïìàààààààààà�_�:��ɺ����Ⱥ���!�-�:�^�X�r����x�n�_���������������ù��'�3�>�'����ܹù������������ܾ����	��&�.�9�?�5�.�"������������	���"�/�5�2�/�.�"���	�����z�u�p�s�z�����������������������z�z�z�z�0�$�!������$�0�=�I�L�X�\�Z�V�I�=�0�	�������������"�.�4�G�T�M�8�.�"�	�����������ûлֻػ׻лû��������������������{�x�v�v�x�z������������������������E�E�EuEgE]EXE\EiEsEuE�E�E�E�E�E�E�E�E�E��Ľ��������½н������(�2�/����н�čā�t�h�[�K�G�O�[�h�tāčĚģħĦĚēč�3�(�(�/�3�?�E�L�Y�e�r�~�������~�r�Y�@�3�ݿѿĿ������������Ŀѿݿ������������ݿ`�^�T�L�K�T�`�m�y�y�������y�r�m�`�`�`�`��ƸƳƣƟƠƤƧƫƳ�����������������������������	��"�.�;�K�N�N�E�;�5�"��	���.�(�"������"�.�.�:�:�3�.�.�.�.�.�.�r�g���������ռ����!�+�.�$����㼘�rƳƧƧƦƧƳ����������������ƳƳƳƳƳƳ������ݿԿѿƿѿݿ������"�����ù����������ùϹܹ�����������ܹϹý̽Ľ̽ݽ����(�4�;�?�9�(�����ݽѽ�����������������������
��
�����������ĳĦĚėčĂąčĚĦĳľ������������ĿĳŇ�{�{�t�o�n�g�n�{ŔŭŹ��������űŠŔŇ�"���"�'�,�4�H�T�a�m�p�v�t�p�e�T�H�;�"��u�p�r�w��������������¼Ӽʼ������������������ľ��������������"�!�����忟���������Ŀѿӿݿ���ݿؿѿĿ�����������������������������������������������������������(�/�4�7�5�4�(��������� ����(�5�Z�g�j�g�c�Z�A�5�(����=�2�9�=�I�V�b�b�b�X�V�N�I�H�=�=�=�=�=�=�ܻѻٻܻ������ �%�'�)�'�������ܻ�ܻл̻ܻ����.�6�@�H�N�@�*�������!������������!�.�9�:�G�Q�G�C�.�)�!�|�x�c�_�W�l�x���������ûлڻڻлû����|�������������������������úɺкɺ������������������������ĽǽĽ������������������<�:�8�8�<�H�U�Y�a�b�a�]�U�H�<�<�<�<�<�<�5�1�/�7�A�Z�s�����������������z�g�N�A�5D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� � w h ' , ` = 5 h u D 7 D C m @ C \ = 5 R M C S D / ' 2 / : ? Q ^ C Y 0 > B S ^ f c L j q b ' n <  1 d N V q  S X U 3 \ 4 i 0  �  �  h  \  �  @  �  b  �  �  $  2      �    X  �  |    4  s  �  �  Z  !  �  |  �  `    �  �  U  !  �  a  g  �  d  �  �  ~  ;  �    c  �  �  �  �  �  n  v  M  �  �  �  i  0  Q  �  �  6;�o<o<o�@��u�'t���`B��C��T����hs�49X�e`B��/��t������D�����ͽP�`��/���P��{��w�\)�+��%�L�ͽ,1�49X��\)��t��Y��}�L�ͽ#�
�D���m�h�,1�����<j�T����7L��o�H�9��7L������-����������t��}󶽃o��q���������-������񪽥�T������������B
�B��B��B�,B~	B1oB�B�jBÕB"�B&t`Bh�B��B ��A���B��B,+B�B?8BoB�B#��BI.B~A���B�BB��B;/B��B4�B��B�4B��B �QB+�B+j�A�3�B	�B3PB-\B'FB�8BX�B&�
B{�B�mB
R�BK�B[2B
��Bw%B��B�KB؆B��B%�B(�B�B��B�tB��BęByCB�B�B��B�qBĖBW�B�B<�B��B4�B!�B&�hBH�BYB �hA��hB�B�B�uB�<Bi�B<�B#�PB��B�pA���BœB��B(SB��BB�B��BAwBF�B O+B+?B+}�A��B	�#BAB-^{BM�B@�B?�B'(;BA�BDSB
ABJ�BDdB
C�BD�B�:B��B�IB�6B%��B)?B�B�B?ABB�B��B��Bɹ@{�VAB�RA3L�A��AAt�QA���A��yA�a�@�l�AI��A���A���A�/@��A��A�mqA�=A���AA��A�I�@mAg>�}A\ jA�j|A�O�B
u�A]�@�MX@��PC��`A.jA�g�?��8Az>Aj8�B�aA^$�A_Z�A ��BцA���>�nZA2gtA��AᇃA��A���@��/A��AyLA�A3��A��BT;@��=@��)A
Ŗ@�j@�jA#�wAĭ~A��C���@t(�AD�A39zA���AtG�A�s�A���A��v@��nAJ��A���A��\A��@��A�J�A�~*A�0�A��lA��qA���A̍Z@g�Q?0�A]�A���A�~%B
E�A\�@� :@�.�C��[A,��A۬2?�%*Ay� AiLB�;A[��A_�<A:�B�A��>�� A3*�A�J�A߫�A��A�t@�"A�BAz�A��A3
	A���BEV@���@�hA
Mj@���@��A#7rAč�A��LC��M   	         4      +               A            <                     5   >         	   &            (   )                        @   	                                   	                     %            0                     "               =            &                     ?   1               #            '                        1                        #                           %                  !                     "               =                                 ?   /               #            #                        +                                                   %                  !   N��N!�PN�O@n�O4��O��N�`�NBkN�xFN�'�P��N)N�N�2�O���Oo�O�hOg�NdKO$ٗNF�P���P;(�OGIO9AO@pOyÊO���N,�O�Og�O�/�N���O���O0!XN�JOk_Oy�-N��P	��Np�FO3"�N�$�O(�:N���O^R�Ox��OoBO��O�=�O:��N��O?O���Nr;OPZ�O�W�O��Oy�O�N*��N���O��N!S    r  �  �    �  �  �  �    �  4  �  �  	�  r    �  {      �  �  V  �  �  H  *  �  P  �  H  n  h    �  4  �  8  �  �  �  .  p  �  o  L  d  �  �  �  �  �    Y  L  �  �  �    �  �  	�  �<D��<49X<49X�49X:�o:�o�D����o��`B��`B��`B�o�t���t���`B�u��C�����1��h��9X��9X��/���ͼ������ͼ�����`B�+��h�+��P�\)�C��t��C���P��w��P�,1���#�
�L�ͽD���<j�<j�L�ͽP�`�]/�L�ͽY��]/�]/�e`B�ixսq���q����C���O߽�C���C���t�����Ƨ�))16BOTOEB:65,)('())���������������������
###
�����������!)5BBHIJFB5)BBNX[\gjjihg[NB>9:=B����������������������������������������GHUabccaUQHFGGGGGGGGhhit����������thhhhh����������������������
#bn{�����{U0
�����

������������#/<GFDHKJHB</*&# ��������������������STV]gmz������zxma\VS��+1)����������������������������������������������:<HSU\XUH<;6::::::::����������������������� ��������������������
O[ZI0
�����>Haz����������zaUPE>�������������PTamnvz�~zmiaXTQLPP����������������������������������������5<IO[ht���{ytWOB6475S[bhituuth[YSSSSSSSSqy��������������tokq�����

�������T[ht����������cYTQOT��������������������������������������������������������������������������������STW_amz�}zxrmaWTSSSY[gt���������tqg[USY����������������������������������������������������������������������NOY[hhplhhijh[ROLMKN9=IMUWZ_bdecbUIC<989�������������������������������������`gt����������tog`]`������������������}���������������������glr�������������ztggGHKQUaknz���zunaRHGGwz�����������zrvwwww���  ������#/<HU\_]XUPHF-# ��������������������#05BHHD@<0&#"SUe{���������wnbULKS������������������������������������������������������������������LN[goskg\[YNFDLLLLLL)16@MSROB6)
:<HMSNH<;9::::::::::�-�+�!����!�#�-�/�:�F�F�S�\�S�F�:�-�-�Z�U�Z�c�f�s�w���������s�f�Z�Z�Z�Z�Z�Z���������!������������g�Z�N�E�A�=�9�@�A�N�Z�g�s�x��}�x�s�k�g���������������������ĿſѿѿǿĿ��������
� ��� �
��/�<�U�a�nÂÌÇ�z�n�a�H��
ŹŷŭŠŗŔœŔřŠŭŹź����������ŹŹ���������������������������������������ؼ�����|�z�t�y���������������������������t�~������������ľ������������������������w�e�G�E�O�h������������������������żŹŸŹ���������������������������������������������������������������������������������$�'�0�'����������������O�B�6������������6�B�X�[�k�n�h�[�O���������������������� �)���������àÚÖÓÍÎÐÓàáìùú������üùìà�����������������������������������������#� ��#�$�/�<�B�>�<�<�/�#�#�#�#�#�#�#�#�H�>�;�/�*� ��"�+�/�;�H�T�Y�\�[�T�T�H�Hà×Þàìùýùïìàààààààààà�_�:��ɺ����Ⱥ���!�-�:�^�X�r����x�n�_���������������ù���,�;�3�'���ܹù����������ܾ����	��&�.�9�?�5�.�"������������	���"�/�3�0�/�-�"���	�����z�u�p�s�z�����������������������z�z�z�z�0�$�!������$�0�=�I�L�X�\�Z�V�I�=�0�	�������������"�.�4�G�T�M�8�.�"�	�û��������»ûлһӻӻлûûûûûûûû����{�x�v�v�x�z������������������������E�E�EuEhE^E\E[E\EiEuE�E�E�E�E�E�E�E�E�E��Ľ������ƽ���������$�,�*������н��O�M�I�O�[�h�tāčĚĜĚėčā�t�h�[�O�O�3�(�(�/�3�?�E�L�Y�e�r�~�������~�r�Y�@�3�ѿĿ��������������Ŀѿݿ���������ݿѿ`�^�T�L�K�T�`�m�y�y�������y�r�m�`�`�`�`��ƻƳƧƥƠƤƧƳ����������������������������������	��"�.�;�G�I�?�;�1���	��������"�+�.�8�6�.�"���������������������ڼ����%�(�����㼽����ƳƧƧƦƧƳ����������������ƳƳƳƳƳƳ������ݿԿѿƿѿݿ������"�����ù¹��������ùϹ۹ݹ�����������ܹϹý�ݽսݽ�������(�4�6�;�5�(����������������������������
��
�����������ĳĦĚėčĂąčĚĦĳľ������������ĿĳŇŁ�~�x�u�zŇŔŠŭŹ��������ŹũŠŔŇ�/�-�.�0�7�H�T�a�l�m�p�r�q�l�a�_�Q�H�;�/��{�u�w����������������������������������������������������������	����Ŀ¿����������������Ŀѿٿݿ��ݿҿѿ�����������������������������������������������������(�/�4�7�5�4�(�������������(�5�Z�g�i�g�a�Z�A�8�5�(���=�2�9�=�I�V�b�b�b�X�V�N�I�H�=�=�=�=�=�=�ܻѻٻܻ������ �%�'�)�'�������ܻ�ܻл̻ܻ����.�6�@�H�N�@�*�����������������������!�.�3�:�7�.�!���x�e�l�x������������ûлٻٻлû�����������������������������úɺкɺ������������������������ĽǽĽ������������������<�:�8�8�<�H�U�Y�a�b�a�]�U�H�<�<�<�<�<�<�5�1�/�7�A�Z�s�����������������z�g�N�A�5D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� � w h  ) ` = 5 h u D 7 D E h > C I = # R M A S D / ' 2 K : @ Q _ C P 0 : ? ? R f c S \ q b ! b "  - d N U q  S V N 3 \ 4 i 0  �  �  h  �  �  @  �  b  �  �  $  2    �  �  �  =  a  |  f  4  s  I  �  0  !  �  |  `  `  �    R  U  �  �  #  �  �  �  �  �     �  �    �  T  '  {  �  �  n  I  M  �  �  =  !  0  Q  �  �  6  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�        �  �  �  �  �  �  u  G    �  �  _     �  �  ^    r  s  t  u  v  w  x  w  t  q  n  k  i  ]  A  %  	   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    I  u  �  �  �  �    b  9  �  �  ?  �  +  s  �  �  �                  �  �  �  �  �  d  6    �  �  j  1  �  �  �  �  �  �  �  c  �  �  �  �  R    �  �  _  #  �  �  �  �  �  x  k  ]  P  B  3  $      �  �  �  �  �  �  �  �  �  �  �  �  ~  w  q  f  Z  N  B  6  *      �  �  �  �  �  �  �  �  �  �  �  ~  v  l  ]  H  ,    �  �  �  U    �  �        �  �  �  �  �  �  �  �  �  �  �  �  |  q  k  d  ]  �  �  �    _  ?  "    �  �  [    �  �  S    �  �  �   �  4  1  /  ,  *  '  %  !            	    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  n  d  Z  D  +     �  �  �  �  �  �  �  �  �  �  �  �  �  x  h  X  E  1  c  �    �  	  	5  	d  	�  	�  	~  	Z  	%  �  �  b  .    �  _  ^    �  �  p  r  q  n  i  _  O  ;  '        �  �  �  �  [  �  r   �  	            �  �  �  �  k  G  !    �  �  �  R    �    9  U  `  l  z  �  �  �  �  }  m  U  -  �  �  �        {  w  s  o  k  g  d  a  ^  [  Y  Z  Z  Z  [  d  o  {  �  �  �  �  �  �  �    �  �  �  �  �  �  X    �  Z  �  j  �  '              �  �  �  �  �  �  �  �  �  �  �  z  c  L  �  �  �  �  f  6  �  �  �  �  M  !  	  �  �  �  f     �    �  �  �  �  �  �  �  e  3  �  �  D  �    �  i  �  �  y   �  V  G  3    �  �  �  �  l  D    �  �  �  �  e  C  ,      �  �  �  �  �  �  �  �  �  �    l  \  N  <  (    �  }    �  �  �  |  y  u  k  b  \  W  X  `  f  f  f  `  V  L  @  5  H  9  2  (        �  �  �  �  h  '  �  }    �  1  �  �  *  "    �  �  �  �  Q  "  �  �  �  r  U  C  -      �  �  s  t  v  x  z  }  �  }  �  }  ^  <    �  �  �  e  
  �  D  P  K  <  (    �  �  �  t  F    �  �  v  <  
  �  �  �  �  �  �  �  �  �  h  ?    �  �  Y  .  �  �  j    z  �  a  Q  '  @  H  F  =  *       �        �  �  h  	  �     l  �  ^  n  n  m  i  Z  F  /              $  1  G  �  :  �  h  d  ]  T  I  :  %    �  �  �  M    �  �  d    �  y  �  
  	  
  
      �  �  �  �  �  �  �    ^  >    �  �  �  �  �  ~  s  d  V  G  7  '      �  �  �  �  �  �  �  h  Q    -  3  0  %      �  �  �  �  c  =    �  �  �  _    �  _  s  �  �  {  o  _  N  _  G  %  �  �  �  I    �  �  �  F  )  .  2  7  4  /  +  #        �  �  �  �  �  �  �  �  �  Z  �  �  �  �  �  k  I  #  �  �  h    �  `    �  �  �  �  �  �  �  �  �  �  �  {  e  O  ;  *          �  �  �  �  �  �  �  �  g  K  2    �  �  �  �  n  S  6    �  ^   �   �  �  �    "  +  .  *    �  �  }  3  �  �     �  �  �  c  �  `  a  \  [  m  ]  I  /    �  �  �  w  K    �  �  �  �  �  �  �  �  �  �  x  j  Z  I  8  '      �        (  3  >  o  ]  J  <  +    �  �  �  k  ?    �  �  G  �  �  A  �  �  !  @  H  K  F  >  4  0  )      �  �  �  :  �  �  ;  �  �  �  
  G  c  _  Q  <     �  �  �  k  '  �  �  "  �  
  V  	     <  Q  w  �  �  �  �  �  h  L  *  �  �  a  �  e  �  X  �  �  �  �  �  �  �  �  �  �  �  �  u  X  1  �  �  �  2    �  �  �  �  �  �  �  �  �  y  j  T  3    �  �  [    �  (  7  �  �  �  �  �  �  M    �  �  �  �  �  �  |  b  G  (    �  �  �  �  �  �  �  �  �  �  s  ]  F  (    �  �  �  �  �  h      �  �  �  �  �  l  ?    �  �  �  Y    �  M  �  �    Y  O  E  :  0  &         �   �   �   �   �   �   �   �   �   �   �  L  9  '    
    	          �  �  �  �  b    �     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  b  3  �  �  n    �  �  �  �  �  �  �  �  n  X  <       �  �  �  O    �  �  W  �  �  `  =  !     �  �  B  �  �  2  �  B  �  �    �  \        �  �  �  �  �  r  M  #  �  �  �  z  Q    �  �  �  �  |  s  j  c  b  `  ^  [  T  N  G  A  <  7  2  /  ,  *  '  �  }  k  X  D  /      �  �  �  �  �  j  N  /    �  �  �  	�  	�  	�  	z  	[  	7  	  �  �  o  &  �  i  �    a  �    �  �  �  �  �  s  [  D  -      
              1  _  �  �