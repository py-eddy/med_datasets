CDF       
      obs    B   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�&�x���       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�`a   max       P�S�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <��       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�=p��
   max       @F�p��
>     
P   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�
=p��    max       @v���Q�     
P  +   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @O�           �  5d   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @��            5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <ě�       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B5I�       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��i   max       B5Hy       9    	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >���   max       C���       :   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =�#m   max       C�8       ;   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       <   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I       =    num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A       >(   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�`a   max       P���       ?0   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�y=�b�   max       ?�͞��%�       @8   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <��       A@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�(�\   max       @F�p��
>     
P  BH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�
=p��    max       @v�z�G�     
P  L�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @N�           �  V�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @�7�           Wl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         GU   max         GU       Xt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�-�qv   max       ?��W���'     �  Y|                        	   *   J   )   \      6      4   �               y            (   9                                    
   L            %   1         ?   H      %                        ,                  
   N(I9N�oN@�N�m]N�]�O�upO�W4N�1�PvCPW��O�p�P���N�tP =�O=E�PB��P8�zN�f�N��COFP]N�!pPuX.O���OȺWO�	8O��O�cNu�N�̞O<C�N�#�N5hPN�MO� M�`aO �O&+�O�w�N(��P�S�NΒ(OK�XO�CyO���O�_N��BO'<P�PSP�Np7
P$��OҠN[c�N),O~N��O,}�Og�NPuTMO	��Oa�N��N�N�N0X�NF�!<��<D��;D��;D��;o�o�o�o�ě��ě��ě��o�o�t��49X�T���T���T���T����C���C���C����
��1��9X��j��j���ͼ��ͼ�`B��h�����o�o�+�C���P����w��w�#�
�,1�,1�49X�8Q�<j�<j�@��@��@��L�ͽL�ͽP�`�P�`�P�`�T���]/�ixս�O߽�hs��hs���P���P���-�������������������������������������������LO[chth`[ONJLLLLLLLL����������������������������������������FOaez�������xmaTKFCF�������#�������# ������������#HU^`^ZTH<#
����������$296)������)6?Oejf[OB6,)-"QZl�������������gWPQBHUY[WacaUHDA?BBBBBB
/<alqrnaU</#
'/<HUaltnnbaUH<8/)&'/Ham{����{aTH;"	���)6BHTSOF)�����()-6BDO[OJB6))(((((([hot����������toh^[[��������������������#)/3<<C<:/##/0%HVmz����zaU<0!����������������������������������������	)3O[diha^[OGB:6)��������������|~����
"%
����������������������������������������������Zanz�������znfa`]^aZIO[ehqqkh_[ZXONIIIII"#/:<D<6/*(#""""""""��������������������jt����������������oj))*67;6)#)))))))))))��������������������#%%)-/263+#��������������������
###"
	���$'"$;4%����� "$(&    ���������������������������������������*6CHNONKC6* ��
#/8<</,#
�����������������������������$)/58BJB5)�������
 .+
���������� )BQ`b`[NB)������GN[\]]][NMJGGGGGGGGG����������������������������������������NO[htjh[OJNNNNNNNNNN6BGNP[d][UNB66666666jt~��������������tjj��������������������W[]got����������og^Wk��������������{qhgk)5B\gq�����tg[H mz~�����������zwifgm)5BO[`gec[NB5)gglrt����ztggggggggg_aknz}znbaa________09<CIILNONKI<700.--0��������������<BFNOV[ONLB><<<<<<<<�����������������������������������������������������¾������������������e�b�^�e�l�r�v�~����~�r�e�e�e�e�e�e�e�e�������!�-�:�C�F�G�F�:�:�-�"�!���U�R�P�R�U�U�a�e�n�q�t�q�n�a�U�U�U�U�U�U������Ϳ˿ѿݿ�����5�H�S�M�A�5��ùïà×Ç�n�b�X�a�n�q�zÅÓìÿ����ÿù�ֺѺѺպֺ���������ֺֺֺֺֺֺֺ־A�:�3�2�2�8�A�M�Z�f��������������s�M�A��ɻ��������������ܼ��1�R�V�U�J�4���F�:�-�.�:�L�x���������������������x�S�Fƚ�u�\�T�6��+�C�hƎ������'�)� ������ƚ�����������������������������������������	������)�6�>�J�o�q�l�p�[�O�6�)�������������������������������������������������������������"�/�C�H�7�4�)�#����H�/�������H�T�a�m�������z�m�a�T�H�/�'�#��"�#�)�/�6�<�@�E�<�<�/�/�/�/�/�/�/�*�#������#�#�/�<�H�T�K�H�A�<�/�/���z�����������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFJFcF�F�F�FxFVFE��x�`�S�I�F�?�@�F�S�_�l�x���������������x�������p�_�[�f�m�y���������Ŀ׿ܿۿѿĿ���������������������������������������������޻�����4�@�Y�s���}�r�Y�M�4�'��@�1������'�4�@�Y��������������f�M�@�`�T�T�P�S�T�`�m�o�n�m�g�`�`�`�`�`�`�`�`�������t�s�r�s���������������������������T�Q�P�S�X�a�i�m�z���������������z�m�a�T��w�{������������������������������U�S�M�U�W�a�k�n�y�n�a�Y�U�U�U�U�U�U�U�U��߼ּԼӼռּ����������������������������ھԾվ����"�0�>�=��	�����������������������������������������һF�=�@�F�M�S�_�l�x������x�l�_�S�F�F�F�F�5�(�#�(�5�A�N�T�Z�g�s���������s�Z�N�A�5�L�D�B�B�L�e�r�~���������������~�r�e�Y�LE7E*E5E7ECECEJEPETEPEEECE7E7E7E7E7E7E7E7�!��
������.�`�y���ݽ���A�E�6����S�!�Z�P�Q�M�G�M�Z�f�s�������s�p�f�Z�Z�Z�ZŹŭťŜœőŔřŠŢůŹ��������������Ź�к��������������ֺ������������п	����������	��"�.�7�:�=�;�5�.��	D�D�D�D�D�D�D�D�D�D�D�D�E
EEED�D�D�D��U�a�n�q�u�t�n�h�a�V�U�S�H�@�<�<�@�H�U�U�0�$����$�*�0�;�=�F�I�L�R�N�M�J�E�=�0�����������ȿѿݿ����������ݿĿ������s�g�g�p���������������������������������������	��"�.�$�"��	����������������ĿİģĿ��������0�@�J�I�?�0��
������Ŀ�������������)�6�B�I�L�B�5�)������ù¹����ùϹԹܹٹϹùùùùùùùùù�ŔŉŇņŇŏŔŠŢŠŞŘŔŔŔŔŔŔŔŔŔŎŇņŁ�{�x�v�{ŇŔŠŨũŦŠŞŚŔŔ�����޹�������� �&�������������V�L�I�F�=�:�3�0�/�0�=�I�V�i�o�t�o�k�b�V�z�q�t�y�������������������������������z·¬©¬©�v²����/�+�-�����·�������������
���#�&�#����
���������B�?�3�+�(�)�#�)�5�B�N�Y�_�_�`�c�_�[�N�B�������������������������Ŀÿ��������Ŀȿ˿ѿѿѿĿĿĿĿĿĿĿĽ�����������(�4�8�9�4�(�&�����ÇÄÇÈÓàì÷ìêàÓÇÇÇÇÇÇÇÇĳĩĦĥĦĳĿ��������Ŀĳĳĳĳĳĳĳĳ J T J K Q b j C ) ! C \ S C J $ / J [ X   : j 8 V Z G j U # D M Y k b 8 e 1 Z k > m W . f 9 h E  b C t M z ` P 6 / 7  L W l Q S )  7  P  k  �  �    k  �  �  �  1  �  �  l  �  4  0  �  
  �  �    �  �  �  �  C  �  �  �  �  ]  $  �  ,  "  �  +  O  �  �  +  �  F  p  $  �  �  m  �  �  �  t  �  ^  �  �  �    �  �  H  U  *  v  ]<ě�<#�
�D���o�T����`B�+�49X�@����
�<j���ͼe`B��%�C���o�����㼓t��o�D���I��0 Ž<j�D���}󶽣�
�o��`B�49X�#�
�'H�9�]/�#�
�L�ͽD���y�#�@���xս<j��O߽�\)���w��j��7L��%��/��F�T����1��+�y�#�Y��m�h��+�����\)������
��Q콗�P���w��^5��-�ȴ9B�CB5I�BA�B,��B��A���B5iB.�B�B{BvB�B��BR2Bf7A���B�BGBm�B��BBB!�2B+Q�B�B��B#|4B"�B��B#nB8�BG�B��BW)Bh�B��B��B!�fB+�B�)BnyBY�B��B/��BQwB!_�B�2B��B!.Bn�B;sB ��BX�B�B
��B�pB
KB
�KB�qA���B��B	�yB��B&[�B��B!_B��B5HyB �B,�OB5�A�� B��BB�BøB�fB��B N�BDAB�NBW�A��iB�B��B=fB@�BHB/~B!��B+PB�#B�qB#��BEiBE�B;�B9vB?�B5B��BC�B��B|pB!��B?�B�oB��B�5B+B/ņB?hB!BB��B� B�cB>�B�B ��BQ(B4dBi!B��B	�XB
�uB��A���BHB	£BɷB&:�B��B8\@���AMu�?���@o`LAƦ�A��Aʱ*@C�PA@_�@�)/@���BA��Aף�A��tA���A���A®fA�}1A���C�3�C���@�]5As7
A��K@�EE@�؏Ai�A�*�A�H�@�l;A�c�A98AX{A��x@���A���?�0�C��vA��AA_�A��
@D��A\)�C�0�A��B
b�A{��A��A��~A�'cA�|E>���A���A�U�?SٷB�A�(A�QA�U�A��"A���Ax��A3��A��A���@���AM	�?���@sAAƀ�A��UA�~�@D)AA �@�>+@��B�WA���A�~�A��QA��A��A���A��AA���C�8=�#m@�ޚAtA���@�ݝ@��Ah��A��A��@���AłpA.�AZ��A�A@�$FA�e?�d�C��A/AC �A�x�@H�0A[�C�/�AŀB
B�A{p�A�y�A�[�A���A�Tn>Q�bA��A�{i?M�8B��A�~�A���A�0DA���A��wAxߧA4�jAʋ�A�i�                         
   *   K   )   ]      7      4   �                y            (   :                        	            
   L            %   1         ?   I      &                        -                                       %   '      #   /   %   =      %      -   -               7      %   %   %   )                     )                  I         !               )   )      -                        5                                       %            -      9            %                  -      %      %                        )                  A                        !   !      %                        3                     N(I9N�oN@�N�m]NEO� �OqCN8�O݂P>�O�IP��QN�tO	�yO1P%E�Ou
N�f�N��COFP]NG��P,ySO��OȺWO.�SO�gO8FxNu�N�̞OȮN�{�N5hPN�MO� M�`aO �O ��O��qN(��P���NΒ(O3��Ov��O���O6�7N��BO�Oғ�OΕENp7
O���OҠN[c�N),O~N��O,}�Og�NPf�:N�A�Oa�N��N�N�h�N0X�NF�!  
   �  �  �  �  {    /  >  ^  �  	�  j  j  �  �  �  \  9  �  	v  �  o  �  6    ?  �  K  L  �  H  L    Z  �  P  {    �  �  z  %  (  ?  �  �    �    �  v  W  �  �  �  �  �  �  k  �  u  �    �  {<��<D��;D��;D���o�D���o���
�T���D����/���
�o��P�u��t������T���T����C���/�#�
��`B��1�������@����ͼ��ͼ��������o�o�+�t������P�`��w�,1�<j�8Q�T���8Q�D���q����hs�@��Y��L�ͽL�ͽP�`�P�`�P�`�T���]/�q����hs��hs��hs���P�������-�������������������������������������������LO[chth`[ONJLLLLLLLL����������������������������������������DHPcgmz������xmaTLGD��������� �������� ���#/HUY[YROH<#
������.74)��������()6BGMOSWVOFBA65)'$(TVam������������z\TTBHUY[WacaUHDA?BBBBBB#/<HMKJHA</# -/<HUahqnjaVUHA</-)- /Ham~��maTH;"��)/6;;5)����()-6BDO[OJB6))(((((([hot����������toh^[[��������������������#/7<<</*#.4HUnz�����znaUH<2,.����������������������������������������-68BIO[ddb_[VOB=960-���������������}��������

 ������������������������������������������������aagnz������zniba_aaJO[chopjh\[[XOOJJJJJ"#/:<D<6/*(#""""""""��������������������jt����������������oj))*67;6)#)))))))))))��������������������#&+/31/)#��������������������
###"
	��"!+)������ "$(&    ����������������������������������������	*6CFLMLHC?6*�����
"!
	���������������������������"),34)�������������	
�����������)5?FIB2�����GN[\]]][NMJGGGGGGGGG����������������������������������������NO[htjh[OJNNNNNNNNNN6BGNP[d][UNB66666666jt~��������������tjj��������������������W[]got����������og^Wk��������������{qhgk)5Bg������t[J;!lmz|������zzmkhillll)5BO[`gec[NB5)gglrt����ztggggggggg_aknz}znbaa________.02<<GIKMONJI<810.-.��������������<BFNOV[ONLB><<<<<<<<�����������������������������������������������������¾������������������e�b�^�e�l�r�v�~����~�r�e�e�e�e�e�e�e�e�������!�-�:�C�F�G�F�:�:�-�"�!���U�T�U�U�X�a�d�n�p�r�n�a�U�U�U�U�U�U�U�U��������ѿҿݿ�����5�F�R�L�A�5�àÜÓÎÀ�n�l�n�zÇÓàìøÿ����ùìà�ֺԺԺֺۺ�������ֺֺֺֺֺֺֺ־Z�M�A�7�6�6�=�A�M�Z�f�s�~����������s�Z�һ����������ܼ��.�E�N�T�R�E�4�����һS�P�G�L�S�_�b�l�x�����������y�x�m�l�_�SƳƚ�u�e�[�O�9�?�\Ǝ��������!������Ƴ����������������������������������������6�5�)���"�&�)�6�B�D�O�W�S�P�O�B�7�6�6����������������������������������������������������������"�5�;�;�1�.�#� ��	���;�9�/�-�'�%�)�/�;�H�T�a�m�q�p�j�a�T�H�;�/�'�#��"�#�)�/�6�<�@�E�<�<�/�/�/�/�/�/�/�*�#������#�#�/�<�H�T�K�H�A�<�/�/���z�����������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFE�E�E�E�FF$F=FVFcFxF�F�F�F�FoFJF$F�x�l�f�Y�S�P�K�S�Y�_�l�n�x�{�����������x�������p�_�[�f�m�y���������Ŀ׿ܿۿѿĿ��������������������������������������������������4�M�Y�l�y�v�f�a�Y�M�4�'��@�?�9�6�=�@�M�Y�f�r�y�����}�r�f�Y�M�@�`�T�T�P�S�T�`�m�o�n�m�g�`�`�`�`�`�`�`�`�������t�s�r�s���������������������������m�k�a�W�Z�]�a�m�z�����������������z�m�m��x�|������������������������������U�S�M�U�W�a�k�n�y�n�a�Y�U�U�U�U�U�U�U�U��߼ּԼӼռּ����������������������������ھԾվ����"�0�>�=��	�����������������������������������������һF�=�@�F�M�S�_�l�x������x�l�_�S�F�F�F�F�5�3�5�7�A�N�[�g�s�v���������s�g�Z�N�A�5�L�E�D�C�D�L�e�r�~�������������~�r�e�Y�LE7E*E5E7ECECEJEPETEPEEECE7E7E7E7E7E7E7E7�!��������.�G�y���н����㽷�y�G�!�Z�P�Q�M�G�M�Z�f�s�������s�p�f�Z�Z�Z�ZŹŭŦŠŞŖœŔŠŭŹ����������������Ź�ֺɺ��������ɺֺ�������������ֿ����������	��"�*�.�4�7�:�;�.�"�D�D�D�D�D�D�D�D�D�D�D�D�EEEED�D�D�D��U�a�n�q�u�t�n�h�a�V�U�S�H�@�<�<�@�H�U�U�$����$�-�0�=�I�L�L�I�I�C�=�0�$�$�$�$���������������ѿݿ����������ݿѿĿ��������w�s�s�w�����������������������������������	��"�.�$�"��	����������������ĿĽıĹ��������(�8�C�A�<�0�#�
������Ŀ�������������)�6�B�I�L�B�5�)������ù¹����ùϹԹܹٹϹùùùùùùùùù�ŔŉŇņŇŏŔŠŢŠŞŘŔŔŔŔŔŔŔŔŔŎŇņŁ�{�x�v�{ŇŔŠŨũŦŠŞŚŔŔ�����޹�������� �&�������������V�L�I�F�=�:�3�0�/�0�=�I�V�i�o�t�o�k�b�V�z�q�t�y�������������������������������z»®¬®¬ ²¿����*�)��
������»�����������������
��
�
� ���������������B�?�3�+�(�)�#�)�5�B�N�Y�_�_�`�c�_�[�N�B�������������������������Ŀÿ��������Ŀȿ˿ѿѿѿĿĿĿĿĿĿĿľ��������������(�4�7�8�4�(�#���ÇÄÇÈÓàì÷ìêàÓÇÇÇÇÇÇÇÇĳĩĦĥĦĳĿ��������Ŀĳĳĳĳĳĳĳĳ J T J K S b \ B 1  S X S $ T $   J [ X 5 / _ 8 6 \  j U  D M Y k b 8 V * Z h > k G / 6 9 I 5 2 b C t M z ` P 6 / 3 X L W l N S )  7  P  k  �  b  �  ;  `       8  �  �  9  i  �  �  �  
  �  g  �  _  �  r  �  |  �  �  ,  �  ]  $  �  ,  "  P    O  n  �  �  �    �  $  >  �  �  �  S  �  t  �  ^  �  �  �  �  �  �  H  U    v  ]  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  GU  
  	  	                �  �  �  �  �  �  �  �  �  �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  e  K  .    �  �  �  �  k  R  9    �  �  �  �  �  �  �  �  �  �  �  �  �  u  d  S  5     �   �  �  �    6  l  �  �  �    /  M  m  �  �  �  �  	   	�  =  �  l  z  v  n  [  :       U  h  \  A    �  �  W  �  j  �  {  �  �  �        �  �  �  b  4  &    3  2  �  �  &  �  �          #  (  -  /  /  $    �  �  �  x  T  2    �  �  �      0  9  +    �  �  �  �  m  7    �  �  x  (  �  @  B  \  V  *  �  �  �  r  R  3    �  i  �  �  v  �  a  �  �      $  &  (  ?  Y  q  �  �  �  �  �  n  D    �  �  _  �  	q  	�  	�  	�  	�  	�  	\  	(  �  �  F  �  �  �  :  �    9  �   �  j  _  T  J  G  F  E  H  M  S  V  X  Z  Z  U  Q  K  B  8  /    &             '  =  Z  h  O  (  �  �  `  2  �    !  �  �  �  �  �  �  �  y  Z  :    �  �  �  @  �  �    S  =  �  �  �  �  �  �  �  �  �  �  �  c  ;    �  g  �  R  f   �  �  �  	�  
�  M  �  U  �  �  �  �  �  *  �    
e  	u  (  �  �  \  Y  V  S  Q  N  L  I  E  B  =  6  /  &        �  �  �  9  8  6  5  3  0  .  +  (  #          �  �  �  �  J    �  �  �  s  b  P  @  -       �  �  �  w  H    �  �  7  �  �  i  �  	  	6  	Z  	s  	�  	�  	�  	�  	~  	w  	m  	b  	U  	F  	  �  �    _  �  �  �  �  }    x  �  C  
�  	�  	d  	  m  �  �  F  z  �       9  O  _  m  n  c  O  2    �  �  f    �  `  	  �  �  �  �  �  �  �  �  �  y  Y  5    �  �    <  �  �  y  "  �  �  �  �    2  6  2  %    �  �  }  B    �  U  �  L   �        �  �  �    B  �  �  �  x  d  w  N  -  �  �  �  �  u  �    �  �  �    3  ?  0    �  �  �  9  �    Q  s  	  �  v  e  R  :  !    �  �  �  g  =    �  �  �  �  �  �  �  K  C  <  4  ,  $    	   �   �   �   �   �   y   r   k   d   ]   V   O  E  =  G  K  J  E  <  4  -  %        �  �  �  t  <  �  �  �  �  �  �  �  �  �  �  |  e  M  5      �  �  �  �  d  @  H  ?  7  /  )  $           X  d  U  F  5  $       �  �  L  )    �  �  �  a  -  �  �  �  z  A    �  �  T      B            �  �  �  �  �  l  F  *    �  �  �  E    u  Z  @  %    �  �  �  �  �  �  �  �  z  p  f  \  R  G  =  3  �  �  �  �  v  ^  F  -    �  �  �  �  V    �  t    �    6  1  ?  O  L  H  @  4  #    �  �  �  �  �  |  j  V  L  W  y  y  o  d  X  L  >  -    �  �  �  �  n  9  �  �  i  	  �    �  �  �  �  \  4    �  �  ~  E  	  �  �  >  �  �  ?  �  �  �  �  �  �  �  �  �  9  �  ~  �  q  �  &    �  l  u  �  �  �  �  �  �  �  �  �  �  �  �  v  c  P  ;  !    �  �  r  c  u  h  U  E  :  %  	  �  �  �  f  %  �  �  6      �    �    #  "      �  �  �  �  �  �  a  4    �  �  A  �  �    &  %          �  �  �  �  �  f  1  �  �  Q  �      
B  
�    >  2    
�  
�  
@  	�  	v  �  i  �  !  g  �  �  �  �  �  �  �  p  K  #  �  �  �  e  '  �  �  A  �  u  �  R  �   �  -  @  }  �  w  b  F     �  �  �  K    �  }  0  �  �  n  |  �  �  �        �  �  �  �  U    �  �  ,  �  4  �  �  b  �  ~    �  |  �  �  �  �  �  �  �  h    �    {  �  �  X      �  �  �  �  �  �  �        -  F  ^  w  �  �  �  �  8  g  �  �  �  v  e  H  "  �  �  �  M    �  "  �  �  )  �  v  R  +     �  �  �  l  m  �  �  �  �  _  *  �  �  {  =  �  W  S  N  F  =  4  (        �  �  �  �  �  �  s  h  f  h  �  �  �  �  �  �  �  �  z  u  r  t  u  v  x  y  z  |  }  ~  �  �  �  �  �  �  �  �  �  ~  t  j  ]  Q  E  9  -  #      �  �  �  �  �  t  4  �  �  a    �  ]     �  D  �  �  <    �  �  �  p  b  T  D  3     
  �  �  �  �  Z  .     �  �  �  �  �  �  ~  k  Y  F  2      �  �  �  �  b  0      �  �  �  �  �  �  �  �  �  �  c  /  �  �  Y  �  �  I  �  �  ~    D  +  %  U  d  W  E  1    �  �  �  �  c  >    �  �  �  #  �  �  �  �  �  �  �  p  [  E  -    �  �  �  \    �  �  �  u  l  b  Y  P  F  =  .    
  �  �  �  �  �  �  �  �  |  m  �  �  �  �  �  �  �  �  u  h  Y  E  2      �  �  A  �  �          �  �  �  �  �  ^  6  	  �  �  o  4  �  �  .  �  �  �  �  �  �  �  �  �  �  �  ~  q  d  W  I  <    �  �  �  {  p  f  [  P  D  3  #      �  �  �  �  �  �  �  t  e  U