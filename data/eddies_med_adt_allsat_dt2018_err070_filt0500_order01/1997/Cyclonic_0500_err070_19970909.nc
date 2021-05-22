CDF       
      obs    H   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��hr�!        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�Qi   max       P���        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��9X   max       <D��        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?
=p��   max       @F�ffffg     @  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @v
=p��     @  ,L   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P�           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @�r`            8   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �	7L   max       ;ě�        9<   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��a   max       B5!        :\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�u	   max       B4α        ;|   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�b�   max       C��p        <�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >g�2   max       C��        =�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          q        >�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          ?        ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          1        A   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�Qi   max       PP��        B<   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Ϫ͞��   max       ?�n��O�<        C\   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��9X   max       <#�
        D|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?
=p��   max       @F�ffffg     @  E�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v
=p��     @  P�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @P�           �  \   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @��@            \�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D
   max         D
        ]�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�ݗ�+j�   max       ?�Q�_p     �  ^�   !                  1            0      q      
                  
            0            
                     	      	                  ?                     R                     5                        #                   -O���N\�O5N1M��N��yO�P8yN��O&]�OCxYOݫ"O2o_P���Ni��O ��N
ʶN�#SO$�JOpN�~O@JVP��ONi_M�QiPP��O��N�+N�k=OE��O��!OqOAs�O�iO�v$N��yN��cO���N\�O��N�l�Nz�"OBo�ORD�P0��N�3KNra%O��N��O�Oh�O��O�\NL$WO�[vN��RO�97O#�xP�sNq�N{��N���OkRO���NƷO.�ZO#�OoZ�N���Ok�6O��kN��OLj<<D��<#�
<#�
;D��:�o��o��o�o��o�ě��49X�D���T����o��C���C���C���t���t����㼣�
��9X�ě��ě��ě����ͼ��ͼ�/��/��`B��h��h��h�o�+�C��\)�\)�\)���#�
�#�
�#�
�'49X�<j�@��D���T���T���Y��Y��]/�]/�]/�ixսixս}�}󶽃o��o��C���C���O߽�\)��hs���㽥�T����罴9X��9X������������������������������������������������������������enpz����zneeeeeeeeeetz��������zwttttttt�������������������)BN]gnkg[N5)
"#&#
 '(%!� �������������������� )6BOVZ[`^RB)!����������������|{#0Icvz}qbU;52,#��*/;<>@=;2/'%********����������������������� ��������������������������������������������������������������������TUabnt{{�{{nbUQOTTTT6BHO]hrtnlh[TOE>><66�������������������������������������S[\chih][[XVSSSSSSSS����#4<:+(�������fs��������������zlefyz����������zsxyyyy6;?HTYZVTOH;81666666ADN[gt�����tg[NMNB>A#'/:HU_eiirnaU<'##�����

�������������������������OOY[htv}wtmljh[XTOMOafgjz}�������zmeaddaNO[hkmnha[OEEGNNNNNN��������������������%)/9BN[jx~}tg[N5'$%
)--)&





)6BO[hlmgb[OB6)$+0<DIPQI<0,(++++++++		����?BHN[_ejqt{tog[B;:<?�����������������������)E: �����������������������������CO[`hjh`[OGB:CCCCCCC�����������������������������������������|vtg_[XTRUX[gitx�������������������������5KMB)�������.0<=IU^a\UUI?<;0&$..������������������������������������������������������������W[ft����������tge^[W�����������������������#/DHU[YQ>#
���,/<=<<:/$',,,,,,,,,,tt}���������tttptttt������������������������������������������$*,,)������� #/6<@EFHH@<<4/+##  RUX^aloz|��zsnaUQQR�����

�����������������������������./49<FHMRTRHB<6/--..��������������������%)BPXWWVXWNCA5-+&a`anz�{znaaaaaaaaaaa���������������������/��
��������������#�/�<�H�S�U�Q�H�<�/�.�&�.�;�?�G�R�T�V�T�G�;�.�.�.�.�.�.�.�.�U�M�H�>�6�<�A�H�U�a�n�q�zÂÅ�z�n�j�a�U�U�N�H�E�E�H�U�V�Y�V�U�U�U�U�U�U�U�U�U�U�M�D�A�?�;�A�M�T�Z�a�f�n�f�Z�M�M�M�M�M�M�N�F�A�=�3�1�3�5�C�N�g�s�����~�s�o�g�Z�N�A�>�5�!���(�5�N�g�s����������s�Z�N�A�ݽսнϽнѽݽ�����ݽݽݽݽݽݽݽݺ�ֺɺĺźɺֺ������!�,�&�!������m�`�T�P�L�P�T�`�m�y�����������������y�m�_�S�:�5�1�3�=�F�S�_�x�����������������_������������������'�)�!���������r�p�{�����о�A�V�^�f�Z�4����ؽĽ��������������������������������������������������������������ʾ׾�������׾ʾ��s�r�j�s�����������s�s�s�s�s�s�s�s�s�s�Ŀ��������������Ŀ̿ѿֿ׿׿ѿʿĿĿĿľ�����	���(�4�A�D�K�G�A�:�4�(�����������������������ʾξо;ʾþ��������������������������ʼʼмӼμʼ����������ɾľʾϾ׾����	���-�"������׾˾ɺ@�6�:�;�3�'� �3�@�L�Y�r���������~�p�L�@��������������������������
�������L�K�L�Y�`�e�i�r�r�r�e�Y�L�L�L�L�L�L�L�L�ú��źźɺ��!�-�:�F�S�h�n�u�l�:����������������������$�*�0�1�1�-�+�$������������������#�(�-�(������A�:�5�,�1�5�A�N�Q�Z�e�_�Z�N�A�A�A�A�A�A���������������������&�&������������������������������	�����������������z�x�m�k�e�p�z������������������������àØÓÑÌÉËÓàìù������������ùìà�ù¹����������ùϹѹܹ����������ܹϹ�����ùìÐÓç��������������������ÇÅÀÄÇÓàéìñïìàÓÇÇÇÇÇÇ�;�:�9�:�;�G�T�\�`�h�c�`�T�G�;�;�;�;�;�;���׾ʾžžʾ׾����	�����������T�N�H�;�5�6�;�H�M�T�a�b�a�^�T�T�T�T�T�T�s�l�g�l�p�w�|�������������������������s���������� �&� �����������ʾɾʾо׾����������׾ʾʾʾʾʾ����	���	��"�/�6�;�@�H�L�O�G�;�/�"��;�1�/�'�)�,�/�;�H�T�a�m�o�u�y�m�a�T�H�;���r�Y�<�3�:�Y�f�s�������������ͼ�������
��������*�5�6�7�6�/�*���������������������������������������������z�a�H�=�B�H�T�a�z���������������������z�\�\�\�h�l�uƁƎƚƧƧƲƧƚƎƁ�u�h�\�\ƎƚƧƬƭƧƢƚƎƁ�u�h�a�Z�\�m�uƁƎƎ���� ��
����5�8�=�A�C�A�A�:�5�(��[�U�T�\�gāĮĿ������������ĿĸĚč�h�[�����������������������������������������������������������������������������I�#��������������#�<�O�c�l�q�m�b�U�I��źŹŭŧŭŶŹ��������������������������ĿĳįĦĦĦĳĿ����������������������ĳĦĚčā�t�i�j�tāčĚĦĪĹļĿ��Ŀĳ���ݿտɿȿѿѿݿ����(�5�B�B�<�5�����[�R�V�[�h�o�t�w�t�h�[�[�[�[�[�[�[�[�[�[�V�U�I�F�=�<�=�I�V�^�b�j�o�o�o�b�V�V�V�V�0�/�$����	�
�����$�%�-�0�7�=�0�0�!����!�.�:�l�y���y�l�e�`�\�S�G�:�.�!�����������������Ľ̽�����ݽнĽ�����FFFFFFF$F1F=F>FJFJFJF=F9F1F%F$FF�������ܾ׾׾߾����	��"�&�)�"��	��D�D�D�D�D�D�EEEE&E*E,E*E&EEEEED��f�d�m�x���������ʼּ޼׼ʼ����������r�fD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�²¦¦¿������������������¿²�_�U�S�I�H�U�n�zÓìùû����ýìàÓ�n�_��ܻۻ׻Իܻ�������������黤�������������������������ûǻʻʻλû� P ^ ) � U #  < n % " : Y D > [ c  ' 6 Z C @ n @ : t A O C 1 @ F e & ! 3 O M 6 ^ ( ! v ; S G v B K l I k I 1 1 W  Q l s k . ` 7 P h < M T ` ;  �  M  ~  s  �    H  <  �  �  �  �  �  �  o  "  �  X  2  �  �  �  �  <  �  P  �  �  �  �  �  �  2  �  �  �  �  �  �  �  �  �  �  �  �  �  �  K  i  �    $  �  �  �    �  ~  !  �  �  �  ;    t  }  B      �  O  ���1;ě��o�D�����
�u�H�9���
�ě����
�m�h�ě��   ����������
�ě��+��h����h�L�ͽ'�����hs�H�9������P�ixսH�9�Y��@��m�h�49X�,1�]/�49X��o�49X�@��y�#�e`B���ixսY���t��aG���O߽���	7L�y�#�u��\)�y�#���P�����xս�C���7L��O߽��w��1�� Ž�E�������ͽ�S��������\�1'B��BW�B!9B"PB��B�B��B8�B@@B*�B͛B�B&0�A��aB^�B�B-��B�B5!B'��BٛB!�qBk�B5vB#MIB �sB HqA�~B	�B�BP�B,GB�AB �B6fB!��Bc�B�=BcxB&3"BCB��B��BK%B�B��BIB�|B	J�B��BM�B&��B�EBB�YB
5cB��B��B_lB
h�B:�B�nB�iBqKB�#B��B,?�B�Bk�B��BѭBcLB>�BBGB!9�B@FB�B��B��B?�BF�B*�3B�B CB'<#A�u	BALB>^B-�+B"�B4αB'�VB�B"�=B,�B>�B"�\B q|B @�A�l�B�4BA�BOyB��B�A�o�B?�B!�B	*�B��B��B&6�B?BD�B�TB?�BMB�1B�BB'IB	?sBAiB@�B&��B��BD@B��B	�WB�rB�B@ B
L�BmBWNB#�B?�BK1B@CB,@B��B��B�lB�B�A�c�Ac�A�`A�b'A=ψA�y+A�bA*��@PT�Ak�W@��A1n�A,��A���AQo{AE�Ay]�A6��ALų@�ߐAW�?��*A�Ⱥ?�L�@f5�B	�A���A��A��A��\A�-�A�U�>�b�A��sA��XAfUAWQ�A��A��1@RAT��A�<�A���@���A�lA�a�A��$B[@BcIA�.�A��A�%VA�l+A�,�A���A�y�A޿qA��}A�s�B˧B	��A��A%�GC��pAY��C�`�@�;0C��A�#�A�Xn@�bA@���A��2Ad��A�rA�,�A=�A�i2A�*A+�@J��Al�@��5A2XA-��A��AS�AD�RAx��A7�AL�@�iAW�1?�qA慂?�p�@b��B�FA�r�A��A�b�A�A�}`A̅>g�2A��A�0Af��AX�FA��A�o�@�r�AS��A�}A�Z@��A��A�}�A��LB<B|kA�u�A�7A�|�A��YA�߄A�z\A㒩A�p�A���AۀVB�B	@�A��A%��C��A[�C�X@�/�C��A���Aǅy@��@�
�   "                  1            0      q      
                  
            0                                 	      
                  ?                      R   	                  5                        $            !      .   #                  #            #      ?                           )         1               !                           #               7         %            '         %            )                                    !   
                                          1                           !         1                                                         1         #                                 #                                    !   
   O��kN\�O5N1M��N��yOx�_Ot�N��O&]�O��O��O2o_P�\Ni��O ��N
ʶN�#SO$�JOpN��O@JVO�}O:��M�QiPP��O�+.N�+N�k=OE��O�ynO�O�N`�O�v$N~8�N��cO���N\�Oz�
N�l�Nz�"O+�O�P��N�3KNra%O�nN��O�OA�EO�*�O�\NL$WO�?�N��O�97N���O�iNq�N{��N���O�O���NƷO.�ZO#�OoZ�N���OW��O��kN��OLj<  �  �  �  {  �  �  �  �  �  #  �  ;  
  �  �  �  U  �  c  )  T  �  x  5  &  �  :  �  �  �  +    �  �  Y  �  �  U  �  �  �  �  �  	�  d  �  �    I  �  �  	  :  �  �    �    C  u  �  x  i  o  g  
s  Q  �  ;  �  �  	�<t�<#�
<#�
;D��:�o�o��9X�o��o�o�C��D���P�`��o��C���C���C���t���t����
���
�����ͼě��ě���/���ͼ�/��/���C��+�t��o�\)�C��\)�\)�'��#�
�,1�0 Ž@��49X�<j�D���D���T���aG���7L�Y��]/�aG��aG��ixսq����\)�}󶽃o��o��O߽�C���O߽�\)��hs���㽥�T���罩�罴9X��9X������������������������������������������������������������enpz����zneeeeeeeeeetz��������zwttttttt��������������������#)5BNQ[^^\[UNB5)#
"#&#
 '(%!� ��������������������!')46BNOPQOEB6))����������������|{#0IUdippcUIC=3+*/;<>@=;2/'%********����������������������� ��������������������������������������������������������������������UUbny{�{ynbURPUUUUUU6BHO]hrtnlh[TOE>><66����������������������������������������S[\chih][[XVSSSSSSSS����#4<:+(�������huz�������������znghyz����������zsxyyyy6;?HTYZVTOH;81666666ADN[gt�����tg[NMNB>A#*/6<HU^dfgfbU<-##�����

��������������������������S[ehjtwtoh`[YRSSSSSSafgjz}�������zmeaddaIOQ[hhjlh[OJIIIIIIII��������������������%)/9BN[jx~}tg[N5'$%
)--)&





')6U[eijhc[OJB7.)$"'+0<DIPQI<0,(++++++++		����@BIN[\cgoskg[NIB<;=@�����������������������)6;1�����������������������������CO[`hjh`[OGB:CCCCCCC�����������������������������������������|vtg_[XTRUX[gitx������������������������#)0*��������.0<=IU^a\UUI?<;0&$..������������������������������������������������������������W[ft����������tge^[W������������������������#4;HQTQF3#
���,/<=<<:/$',,,,,,,,,,tt}���������tttptttt������������������������������������������$*,,)������� #/6<@EFHH@<<4/+##  RUX^aloz|��zsnaUQQR�����

�����������������������������./49<FHMRTRHB<6/--..��������������������%)BPXWWVXWNCA5-+&a`anz�{znaaaaaaaaaaa���������������������
����������������#�/�<�H�N�R�H�<�/��
�.�&�.�;�?�G�R�T�V�T�G�;�.�.�.�.�.�.�.�.�U�M�H�>�6�<�A�H�U�a�n�q�zÂÅ�z�n�j�a�U�U�N�H�E�E�H�U�V�Y�V�U�U�U�U�U�U�U�U�U�U�M�D�A�?�;�A�M�T�Z�a�f�n�f�Z�M�M�M�M�M�M�N�I�A�?�5�3�5�A�J�N�]�g�q�����|�s�g�Z�N�A�:�.�-�3�5�A�N�Z�g�j�s�x�z�z�s�g�Z�N�A�ݽսнϽнѽݽ�����ݽݽݽݽݽݽݽݺ�ֺɺĺźɺֺ������!�,�&�!������m�l�`�[�T�Q�N�T�T�`�m�y�~�����������y�m�l�_�S�O�F�E�F�S�W�_�l�x�����������{�x�l������������������'�)�!�����������������������Ľݾ�4�D�G�A�4���ݽ��������������������������������������������������������������ʾ׾�������׾ʾ��s�r�j�s�����������s�s�s�s�s�s�s�s�s�s�Ŀ��������������Ŀ̿ѿֿ׿׿ѿʿĿĿĿľ�����	���(�4�A�D�K�G�A�:�4�(�����������������������ʾξо;ʾþ����������������������ƼʼϼҼͼʼ��������������ɾľʾϾ׾����	���-�"������׾˾ɺ@�=�D�B�@�8�@�L�r�����������~�r�e�Y�L�@��������������������������
�������L�K�L�Y�`�e�i�r�r�r�e�Y�L�L�L�L�L�L�L�L�ú��źźɺ��!�-�:�F�S�h�n�u�l�:��������������������������$�0�0�-�)�$������������������#�(�-�(������A�:�5�,�1�5�A�N�Q�Z�e�_�Z�N�A�A�A�A�A�A���������������������&�&����������������������������������	�������������������z�t�m�l�m�y����������������������àÛÔÓÒÍÑÓàåìû����������ùìà�ù����������ùǹϹٹܹ�ܹϹùùùùù�����ùìÐÓç��������������������ÓÈÇÃÆÇÓàåìéàÓÓÓÓÓÓÓÓ�;�:�9�:�;�G�T�\�`�h�c�`�T�G�;�;�;�;�;�;���׾ʾžžʾ׾����	�����������T�N�H�;�5�6�;�H�M�T�a�b�a�^�T�T�T�T�T�T�s�r�q���������������������������������s���������� �&� �����������ʾɾʾо׾����������׾ʾʾʾʾʾ����	���	��!�/�;�H�J�M�J�H�E�;�/�"��T�O�H�;�/�+�,�/�1�;�H�T�a�g�m�p�s�m�a�T���r�f�>�6�=�M�Y�f�p�������ڼ�ؼʼ�������
��������*�5�6�7�6�/�*�����������������������������������������������z�a�H�?�@�D�I�T�a�m�z�����������������\�\�\�h�l�uƁƎƚƧƧƲƧƚƎƁ�u�h�\�\ƎƚƧƬƭƧƢƚƎƁ�u�h�a�Z�\�m�uƁƎƎ�����������(�5�9�A�?�@�9�5�(��[�b�l�{čĚĦĳĿ��������ĳĪĚā�t�h�[�����������������������������������������������������������������������������U�I�)�#��� � �
��#�<�L�U�`�j�n�k�b�U��żŹŮŷŹ������������������������������ĿĳįĦĦĦĳĿ�����������������������t�k�l�tāčĔĚĥĦĳĴĳıĦĚčā�t�t�����ڿοϿݿ������(�5�<�=�7����[�R�V�[�h�o�t�w�t�h�[�[�[�[�[�[�[�[�[�[�V�U�I�F�=�<�=�I�V�^�b�j�o�o�o�b�V�V�V�V�0�/�$����	�
�����$�%�-�0�7�=�0�0�!����!�.�:�G�S�`�k�b�`�Z�S�G�:�.�!�!�����������������Ľ̽�����ݽнĽ�����FFFFFFF$F1F=F>FJFJFJF=F9F1F%F$FF�������ܾ׾׾߾����	��"�&�)�"��	��D�D�D�D�D�D�EEEE&E*E,E*E&EEEEED��f�d�m�x���������ʼּ޼׼ʼ����������r�fD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�²¦ ¦²¿��������������������¿²�_�U�S�I�H�U�n�zÓìùû����ýìàÓ�n�_��ܻۻ׻Իܻ�������������黤�������������������������ûǻʻʻλû� P ^ ) � U   < n '  : f D > [ c  ' 2 Z R = n @ ; t A O 6 / @ 4 e # ! 3 O C 6 ^ * , s ; S J v B > R I k A 0 1 N  Q l s F . ` 7 P h < @ T ` ;  �  M  ~  s  �  �  �  <  �  Q  R  �  K  �  o  "  �  X  2  �  �  �  �  <  �    �  �  �  y  e  V  u  �  �  �  �  �    �  �  s  H  �  �  �  �  K  i  �  �  $  �  c  �         !  �  �  A  ;    t  }  B    �  �  O  �  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  D
  �  �  �  �  �  �  �  �  �  �  x  P    �  p    �  �  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �       	        �  �  �  �  �  t  c  L  )  �  �  �  |  F    �  �    �  J  {  �  �  �  �  �  �  �  �  �  �  �    
        �  �  �  �  �  �  �  �  �  v  j  ^  Q  B  0      �  �  �  v  R  .  �  �  �  �  �  �  �  �  �  b  A    �  �  �  �  �  }  _  E  �  1  ]  t  �  �  �  �  �  �  �  �  }  =  �  �  '  �  �  g  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  n  a  T  G  :  �  �  �  w  k  ^  M  ;  )    �  �  �  �  q  K  2       J         !        �  �  �  �  �  o  I    �  �  w  4   �    ?  R  U  Z  g  r  {  �  �  �  �  n  D    �  e  �  p    ;  4  *      �  �  �  �  �  �  �  e  B    �  �    I    �  w  �  	[  	�  	�  
  
  	�  	�  	]  �  {  �  �  g  �    F  -  �  �  �  �  �  �  �  �  �    t  j  _  `  y  �  �  �  �  �  �  �  �  �  �  r  ]  H  2         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  W  <  !     �   �   �   �   �   x   a  U  M  E  >  6  /  )  %  !      	    �  �  �  �  �  �  �  �  �  �  �  �  �  q  X  A  )    �  �  �  �  �  d  8  �  �  c  b  `  [  V  Q  G  ;  )    �  �  �  �  �  }  _  >     �    !  (  #          �  �  �  �  �  g  <    �  �  t  =  T  D  3  "    
      	    �  �  �  �  �  �  �  u  V  6  y  �  �  �  �  �  �  �  �  �  �  �  m  ;    �  �  D    �  o  w  s  g  U  A  )  
  �  �  �  w  X  ;  !    �  �  �  D  5  3  0  .  ,  *  (  %  #  !  !  "  $  %  &  (  )  +  ,  .  &  	  �  �  �  �  �  �  �  �  �  �  �  `  -  �  �  -  �  %  �  �  �  �  �  }  c  E  $    �  �  y  4  �  �  R    �  g  :  K  ]  o  {  r  j  a  V  I  <  /  
  �  �  f  <     �   �  �  �  �  �  �  �  �  }  u  l  b  W  K  @  4  +  #      	  �  �  �  �  �  �  �  �  �  w  k  c  v  �  �  �  �  �  �  H  �  �  �  �  �  �  �  �  h  K  (  �  �  �  w  K    �  L  �  �  �      (  *  $    �  �  �  �  �  f  >    �      �  �  �           �  �  �  v  E    �  �  �  N  �  �    �  �  �    s  e  p  �  �  �  �  �  �  t  b  O  <  (    �  �  �  �  �  �  �  �  �  l  P  *  �  �  M    �  P  �  n  �  �  /  A  Q  W  W  T  M  D  4  #    �  �  �  �  �  �  f  M  3  �  �  �  �  y  r  m  g  c  ^  Y  T  O  G  ?  /      �  �  �  �  �  �  �  �  �  n  S  3    �  �  �  �  [    �  {  $  U  K  A  2  #      �  �  �  �  �  u  S  2    �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  S  8    �  �    �       �  �  �  �  �  �  }  w  r  m  i  e  a  [  Q  H  5  �  �  o  �  �  �  n  S  8    �  �  �  �  `  7    �  �  �  r  [  D  �  �  �  �  �  �  �  �  �  _  4    �  �  M    �  �  H  �  �  �  �  �  �  �  �  �  �  �  �  z  t  m  f  ^  P  :      	�  	�  	�  	�  	�  	�  	q  	8  �  �  J  �  �    �  �  ;  c  k  g  d  E  &    �  �  �  �  �  �  �  d  E  #  �  �  �  i    �  �  �  t  \  G  1    �  �  �  �  p  ?    �  �  m  3   �   �  �  �  �  �  �  l  O  /    �  �  �  �  W      �  g  �  q      �  �  �  �  �                        �  �  I  3  $       �  �  �  �  y  T  2    �  �  s  &  �  D  �  �  �  �  �  �  x  ^  ;    �  �    <  �  �    �    �    �    H  �  �  �  c  ;    �  �  F  �  '  
z  	�  �  �  �  �  	       �  �  �  �  �  �  �  �  �  �  �  �  p  P     �   �  :  8  5  3  &    	  �  �  �  �  �  �  �  n  U  ;    �  �  }  �  �  }  o  a  Q  @  (    �  �  �  `  D  !  �  �  C   �  �  �  �  �  �  �  �  �  �  }  h  S  8    �  �  �  w      �      �  �  �  �  �  �  �  �  �  d  >    �  �  v  3  �  �    �  �  �  �  |  i  R  7    �  �  �  E  �  �  2  �  E   �  �  �      �  �  �  �  �  �  �  ]  -  �  �  "  �  �      C  L  V  _  `  `  _  7  �  �  �  t  V  7    �  �  �  �  m  u  k  `  V  L  A  7  *        �  �  �  �  �  p  O  .    �  �  �  �  �  �  �  �  z  a  G  .    �  �  �  n  G  !   �  L  c  w  l  _  J  3    �  �  �  �  �  g  H  )    �    @  i  a  Y  P  F  :  ,      �  �  �  �  s  L  "  �  �  u    o  W  :      �  �  �  �  �  �  ^    �  @  �  o  �  u  �  g  Q  <  $    �  �  �  �  U  "  �  �  r  /  �  �  v  h  S  
s  
F  

  	�  	�  	/  �  y    �  D  �  E  �  ~  �  '  g  �  �  Q  4    �  �  �  �  �  j  G  #  �  �  �  d    �  /  �   �  �  �  �  �    N    �  w  (  �  |    �  >  �  @  �  $  �  �  8  "    �  �  �  U    �  �  2  �  �  $  �  e  �  ~   �  �  �  h  Y  D  4  4  7  #  �  �  {  0  �  �  ,  �  t    }  �  q  F  !    �  �  �  �  �  �  j  R  9    �  �  �  �  S  	�  	�  	�  	�  	z  	q  	M  	  �  �  D  �  �    �  +  �    i  �