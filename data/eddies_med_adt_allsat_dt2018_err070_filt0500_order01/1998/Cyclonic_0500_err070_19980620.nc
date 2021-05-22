CDF       
      obs    E   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�n��O�<       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P`f5       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �� �   max       <T��       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?.z�G�   max       @F�=p��
     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v�
=p��     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @2         max       @P�           �  6x   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�@            7   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �1'   max       <o       8   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��m   max       B48�       9,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�V   max       B4|�       :@   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�:%   max       C���       ;T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�u�   max       C��t       <h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          H       =|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3       >�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -       ?�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P��       @�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��i�B��   max       ?��Q�`       A�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �� �   max       <T��       B�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?.z�G�   max       @F�=p��
     
�  C�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v�
=p��     
�  N�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P            �  Y�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�Y            Z   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E&   max         E&       [$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�N;�5�Y   max       ?�ح��U�     �  \8                           )                        H         %   !         
      -                                              $   "   2      4                        $               6   $         "            =         %N2��N$��N$�NJ�vOA4�N$��N_�AO3V�P(�N�	�O(|TOZ*O�:N��O��ODXCP`f5O{e+OQ�P9NO�H�N�S;Of	�O��O��P-�~O�)�O�`�O�߸NP��O�ޝO��N�X{O(e�Nx�N��eNI%:N�N	N��/OF�O�E�P\jON��P �N���M��O�_N�aN�O<DM���Oj�O��>O<\N��O�P#hhO��<On(�O���O���N*�@N��LNF��O|�QN��5N���O�<T��<49X<#�
;�o��o�D���ě��t��t��t��D���D���T���e`B�e`B�u�u��C���C����
���
��1��j��j��j�ě��ě��ě��ě����ͼ�/��/��/��`B�o�\)�\)�t��t���P��P���#�
�'',1�,1�49X�8Q�8Q�<j�@��D���D���P�`�Y��]/�e`B�u�u��%�����7L��C���C���t���1��{�� �������������������##/<DHJNH</#########

#%/'#







(���������������z|���$ ��������������������JN[gt�������tg\QNKJJ��������
$"��������������������������������
 +**#
�����#/<BHNJH<;1/#"��������������������<</'(&#$/<<CHOHA<<<<)5BUZ]^[N5)��������������������0IUbgmtrnU<63$������������������#/=HNLHA<3/#
����������
����������������
�������X[ht���{th[UXXXXXXXX���������������26BO[fhmih\[OB>61/22t���������������tltt)Ngfkr|}mmg[P)GKanz��������znaUMHGsz�����������znebgis����������������������������������������`hmz����������zmaWV`�������������������������������������{������������������������������������������������������������������������36BMOOPROBB<64333333MOQP[\hhmjhe[POLMMMM�������

���������������������������BN[tv��������tgWF<<B_fmz�������zmaWUTX[_V[_gtw������|tig[ZVV������++%���������������������������

./2;HTailmljaTOH;51.[[]htxxth[[[[[[[[[[[;<ACIUZUUOI<;;;;;;;;��������������������������������������������������������~ln{���������{vsjllpl��������������������gnz�������znlaggggggfgt����������tgg]_ff���*AJ?)�������5BNVOLB5)���Z^gt������|tsg`^[YXZ)5[bgjnpsg[B5����������������������������������������~��������������~~~~~#*06620'#��
#&+/6>></#
���NNB66))/6DO[aehhh][N:<>HIU]_\UUTKHC<:8::UPHE</###&*//8<GHMSU�������������������������������������������������������������������������������伱���������������������������������������4�+�(���(�4�4�<�A�A�A�4�4�4�4�4�4�4�4���������������������������������к3�,�,�3�@�D�L�M�L�@�3�3�3�3�3�3�3�3�3�3�h�]�b�h�t�zāćĆā�w�t�h�h�h�h�h�h�h�h�׾ξ־�������	������	�������׿	�����Ӿ����������ʾ׾��.�T�W�;��	�U�R�H�B�<�;�4�<�H�K�U�a�h�j�d�b�a�W�U�UàÞÓÇÄÇÌËÓàìù������ýùòìà�H�A�=�>�H�S�U�a�c�m�n�z�|Á�z�n�a�U�H�H�z�p�n�k�m�n�y�zÇÐÓßàáæàÓÇ�z�z�t�t�y�t�g�[�N�B�7�B�B�N�U�[�_�g�t�t�t�t�i�m�v�z�������������ǿͿȿ����������y�i�����������Ľͽнݽ�����������ݽĽ������~�|�~��������A�E�K�V�M�4���н����Z�M�A�(�������(�4�G�M�X�]�Z�[�^�Z�O�6�)�(�&�)�)�.�6�B�I�[�d�h�j�`�f�_�[�O����ŹŠŇ�n�f�rŎŠ����������������O�N�@�*�������������*�C�T�R�^�\�V�O���������������������������������������������������.�;�M�T�\�T�G�;�.�"��"�����"�&�.�;�=�G�H�I�K�G�C�;�.�"�"���׾����������������پ��	�����	���������������6�O�h�uƅƄ�h�N�C�6��������s�O�B�?�K�N�Z�g�s���������������������������������������	��"�5�>�?�;�/�"�	���s�j�N�5������5�P�g�n�v�����������sǔǊǈǂǈǔǡǦǭǡǔǔǔǔǔǔǔǔǔǔƧƚƎ�z�j�^�a�uƎƧƳ����������������Ƨ����������������$�0�3�8�7�4�0�$������ʾɾ¾��������Ⱦʾ׾޾������׾ʾʾ�{�~����������������ž¾������������ìêêëìñù��������ùìììììììì���������~�y�n�y�������������������������G�D�;�7�5�8�;�E�G�M�S�T�V�T�G�G�G�G�G�G�S�L�H�S�\�_�l�x���y�x�l�c�_�S�S�S�S�S�S�������������������������������������������ݿѿ̿ǿƿ˿ѿݿ������ ��������3�'������3�L�Y�e���������~�r�Y�@�3����������������������&�)�(��������ĳĦęĚĩĿ���������������������Ŀĳ���������������~�����������������������������������ʼݼ��!�.�<�<�4����ּ����������������������������������������������$��!�$�0�=�=�=�5�0�$�$�$�$�$�$�$�$�$�$�s�Z�N�@�5�1�/�5�A�N�Z�g���������������s�'�#�����'�*�,�(�'�'�'�'�'�'�'�'�'�'�����������������������������������������g�N�K�G�N�Z�]�g�s�������������������s�g������������� �����������������������)������������)�6�B�M�T�R�O�F�B�6�)���������4�M�f�q�u����r�Y�@�4������������������$�%�-�0�3�2�0�+�$�����������������ĿпѿѿѿѿѿĿ������������{�x�s�s�n�c�n�{ŇŔśŠŨŦŠŠŔŇ�{�{�l�L�8�.�.�4�G�V�y���Ľн��ֽĽ������lÓÇ�z�n�[�R�O�O�U�a�nÏëöùû÷ìàÓ¿²¥¦¿���������
�
������������¿ĚĖĜěĠĦĭĳĿ������������ĿĵĳĦĚ���l�T�:�F�S�_�l�������ûлֻٻѻû�������������������������������������� �
����#�%�#�"���
���������û��ûȻлܻ����ܻлûûûûûûû�EEEEEEEE$E7ECEPESE^E`EZERECE7E*E����������ù÷ìèàÛÙàì÷ùù������F$F#F$F.F1F:F=FJFVFcFcFcFVFSFJFAF=F1F$F$D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� % � [ X ? : R g T Q . . $ J S 5 P A 2 O _ d M ; c D ( Y x L J G I & Z [ q c : 6 L M 5 q j b B K @ Y 7 i J Q 2 U 2 S I T q � B F : < T j 7    I  �  P  u  �  <  �  �  M  .  q  *  3  �  �  �  )  ,  �  �  �  �  '  J  �  '  }    N  V  G  �    c  S  �  �  �  �  ^  �  �  �    6  �  *  �  =  K  �  :  �  �  �  �  E  U      �  �  S  �  g  �  2  �  [%   <o;��
;o����t��D����1�H�9��/�������ͼ�h��1�o��󶽰 ŽC��#�
�ixսY��o�0 Žo��㽋C��H�9�'e`B��P�T���T���C��P�`�t��#�
�'@��Y��m�h��t���t���Q�@���j�8Q�<j��hs�L�ͽD���m�h��o��1��%�y�#�y�#��+��;d�ě����P��^5�ȴ9��hs��\)�����1'������^5���mB�B��B$��B�NB��BboB�UB	�'B�;B ��B�B4�BM�BsJB��B"��B&��B�gB�B��B�PB�<B�qB�YB �B��B��BJ�B��B"B 6Bw�B48�B��B5�B*U6B,u�B>�BNzB��Bi�B	��A�EIB	�dB-��BJ�B��A��mB�B&�:BēB�B@zB)C�B�2B^�B	ɬBV�B}B	�B�kB�kBA�B
�B%n�B]B�KBPLB�B��B��B$��B=iB~�BB�B��B	�B�8B ��B?�B26B@8B��BےB"�B&�?B��B8gB��B��BÞBB�BNpB��BMB�Bo�BB1�B �BE�B4|�B�BŁB*1�B,F�BDRB?�B�B@�B	�A���B	�PB.49B5�B0�A�VBɐB&�B��B�*B@B)a�B0�B`�B	��B�iBAB	��BCB�BF�B+�B%B�B@UB��B�|B��A��vA�{`@��A8a�A�V�?�}�A�H\AXݦAX��A�N�A�S{A��A�y.A�d�AryA)v�A+n"A8�TA��A���A�l@�2XA_CiAa*zAT�BA���A��A��A�XFB`UB��B�AR��AIcA�=#AqL>Ad��@���@�WIA~`?�'%A�nA�KA��A�OA�;�B
�A���?�:%A���A��FB7�A�WF@�p�B	�Ax^A�qA�:A�K[A��A�i@�ļ@^r>A趇@���C��
A��LC���C��A���A���@� kA74A�~�?�HA܀aAY<,AT��A�x�ÀAŃ�A�e+A��FAqP5A)�A(�7A8��A�k�A��ZA�D@�PA`�AaQ�ATn�A�qHA��bA�]PA�|]B@B:�B	@ARAH��A�t�Ar�}Ad��@�0@�u�A}�?�5�A��`A�|�A�h�A�A�"�B
5�A�o;?�u�A��A��xB?�Aց�@�̿B�[Ay.�A�r3AF0A�z�A� �A�|u@���@\|�A��@���C��5A�~3C��tC��                           )                        H         &   "         
      -         !                                    $   #   3      4                        %         	      6   %         "            >         &                           3                  !      3         1   #            %   -      #   %      #                              #   )         1                           %            1   #         !                                                +                        )         +                        !         #                              #            -                           #            )                                 N2��N$��N$�NJ�vNY��N$��N_�AN�L�P��N�7�OWN���N�W�N��O�s�O"6�PyhOI�O�_P��O{W|N�S;O��O��O���OP�,O��O�|N��INP��O�ޝO{fN�X{Nѭ�Nx�N��eNI%:N�N	N���OF�O�E�O�(O�)N��PntN���M��O�_N�aN�O<DM���OBO�O�ȎO<\N��N���O�f?O��On(�O���O"U�N*�@N��LN��O#�N��5N���O�    N  B  q  �  �  �  �  U    w  �  9  �  �  e  �  �  ]    �  �  �  �  �  �  �  v  d     T  �  8  �  [  4  �  �  Q    �  �  |    �  �  /  �  4  ]  �  �  �  �  �  g    4  �  
  O  �    �  �    �  �  	�<T��<49X<#�
;�o�#�
�D���ě��D���u�49X�T���u��C��e`B��o��C���P���㼴9X���ͼ��ͼ�1��`B��j�ě��8Q��/��`B��w���ͼ�/����/�C��o�\)�\)�t���P��P��P�<j�<j�'49X�,1�,1�49X�8Q�8Q�<j�@��T���H�9�P�`�Y��m�h�����o�u��%��t���7L��C���O߽� Ž�1��{�� �������������������##/<DHJNH</#########

#%/'#







(��������������������$ ��������������������P[gt|������tg`[TPPPP�������

������������������������������� 
#)()#
���� #/<<DB<6/#       ��������������������<</'(&#$/<<CHOHA<<<< )5BRWZ[NB5)	 ��������������������#0<IUZ_egheZUI<2' #��������������������#&/<GHB></*#��������� ���������������
���������X[ht���{th[UXXXXXXXX������������������26BO[fhmih\[OB>61/22�����������������yv�%)5BJNT[][TNB5,)"%Tanz������zna\ULKNTmrz�������������znmm����������������������������������������`hmz����������zmaWV`�������������������������������������{������� ���������������������������������������������������������������������36BMOOPROBB<64333333MOR[hhlihd[QOMMMMMMM�������

���������������������������ENgt��������tg[TMJDE^bjmz������zma\XXZ^V[_gtw������|tig[ZVV�����*+)#���������������������������

./2;HTailmljaTOH;51.[[]htxxth[[[[[[[[[[[;<ACIUZUUOI<;;;;;;;;����������������������������������������������������������nootz����������{wtkn��������������������gnz�������znlaggggggcgrtx����tqgccccccc��&6<;6)������)5CJHB<5)����Z^gt������|tsg`^[YXZ)5[bgjnpsg[B5����������������������������������������~��������������~~~~~"#&04400.#""""""""��
#+/067/#
 �NNB66))/6DO[aehhh][N:<>HIU]_\UUTKHC<:8::UPHE</###&*//8<GHMSU�������������������������������������������������������������������������������伱���������������������������������������4�+�(���(�4�4�<�A�A�A�4�4�4�4�4�4�4�4��������������� ���������������������3�,�,�3�@�D�L�M�L�@�3�3�3�3�3�3�3�3�3�3�h�]�b�h�t�zāćĆā�w�t�h�h�h�h�h�h�h�h�����������	�
����	������������ھǾ��������ʾ׾���	�.�E�J�;�"��H�D�=�<�7�<�H�N�U�a�e�g�a�a�a�U�H�H�H�Hìåà×ÓÎÍÓÙàìù��������ûùïì�H�C�A�G�H�U�a�f�n�v�u�n�a�U�H�H�H�H�H�H�z�u�o�s�zÇÓÙÜÜÓÇ�z�z�z�z�z�z�z�z�t�t�y�t�g�[�N�B�7�B�B�N�U�[�_�g�t�t�t�t�y�t�r�w�|�����������¿ſʿſ����������y�����������Ľнݽ������������ݽнĽ������������������н��4�8�<�-��ݽ������4�(����
����(�4�@�M�V�[�Z�V�M�A�4�B�7�6�)�(�)�,�.�6�B�O�[�`�^�[�Z�_�[�O�B����ŹŠŇ�v�yœŠ�����������������������������
���*�@�I�O�O�H�C�6�*�������������������������������������������	��������	��.�;�D�G�J�G�A�;�.�"��"�����"�&�.�;�=�G�H�I�K�G�C�;�.�"�"�޾׾ʾ����������������վ�����	������������*�6�C�O�R�N�H�C�@�6�*���g�R�E�C�N�Z�g�s���������������������s�g�	�����������������������	��"�*�0�*�"�	�����(�5�A�N�R�O�N�A�5�(������ǔǊǈǂǈǔǡǦǭǡǔǔǔǔǔǔǔǔǔǔƧƚƎ�z�j�^�a�uƎƧƳ����������������Ƨ����������������$�0�1�4�6�5�2�0�$�����ʾɾ¾��������Ⱦʾ׾޾������׾ʾʾ�����~�������������������������������ìêêëìñù��������ùìììììììì���������~�y�n�y�������������������������G�D�;�7�5�8�;�E�G�M�S�T�V�T�G�G�G�G�G�G�S�L�H�S�\�_�l�x���y�x�l�c�_�S�S�S�S�S�S�������������������������������������������ݿѿ̿ǿƿ˿ѿݿ������ ��������3�'������3�L�Y�e���������~�r�Y�@�3��������������������� ����������ĿĳĦĠğĦĭĿ�������������������Ŀ���������������~�����������������������������������߼���!�.�:�:�.�����ּ��������������������������������������������$��!�$�0�=�=�=�5�0�$�$�$�$�$�$�$�$�$�$�s�Z�N�@�5�1�/�5�A�N�Z�g���������������s�'�#�����'�*�,�(�'�'�'�'�'�'�'�'�'�'�����������������������������������������g�N�K�G�N�Z�]�g�s�������������������s�g������������� �����������������������6�)���
���������)�6�B�J�Q�O�B�A�6�4�'���������4�@�Y�f�m�r���{�r�Y�@�4��������������$�%�-�0�3�2�0�+�$�����������������ĿпѿѿѿѿѿĿ�����������Ňŀ�{�z�{ŀŇŔŠŠŢŠŚŔŇŇŇŇŇŇ�S�D�9�F�a�y�����Ľ˽ֽ׽˽Ľ��������l�S�z�r�n�^�Y�V�U�a�nÇÓâîõöòàÓÇ�z¿²¥¦¿���������
�
������������¿ĚĖĜěĠĦĭĳĿ������������ĿĵĳĦĚ�n�q�x���������û̻лԻл̻û����������n��������������������������������� �
����#�%�#�"���
���������û»ûʻлܻ����ܻлûûûûûûû�E*E(EEEEE E*E4E7ECEPEWE[EVEPEMECE7E*����������ù÷ìèàÛÙàì÷ùù������F$F#F$F.F1F:F=FJFVFcFcFcFVFSFJFAF=F1F$F$D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� % � [ X X : R W U H * ,  J K 1 [ 0 5 P < d = ; Y / , R D L J < I  Z [ q c / 6 L H 0 q g b B K @ Y 7 i < I 2 U $ Q @ T q f B F 9 A T j 7    I  �  P  u  k  <  �  >  �  �  :  �  �  �  �  ^  �  �  8  �    �  l  J  d  �  $  J  �  V  G  $    �  S  �  �  �  }  ^  �  �  a      �  *  �  =  K  �  :  �  �  �  �  �  �  x    �  �  S  �  G  _  2  �  [  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&  E&    )  =  K  H  @  1    �  �  �  {  A    �  �  F     �  o  N  Q  T  X  [  ^  a  c  c  d  d  e  e  d  _  Z  V  Q  L  G  B  6  )        �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  k  e  `  Z  U  O  I  D  >  8  2  +  %               g  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ^  1  �  �  �  �  �  �  �  �  �  �  �  �  x  p  e  T  C  0    �  �  �  }  x  r  f  W  I  8  &    �  �  �  �  �  c  ?     �   �  �  �  �  �  �  �  �  �  �  |  m  ^  L  5    �  �  �  4  �  7  I  P  S  H  2    �  �  �  �  l  7    �  �  �  =  �  z  �  �        �  �  �  �  �  �  �  y  t  q  e  Y  L  >  2  l  t  t  k  Z  F  1    �  �  �  �  �  |  l  Q  C  8  -  #  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  <    �  �      &  -  4  8  8  4  0  *  !    �  �  �  a  #  �  �  R  �  �  �  �  �  �  �  �  �  {  m  `  \  Y  J  .    �  �  �  �  �  �  �  �  �  �  o  X  ?  $    �  �  �  �  G  �  �  g  [  c  e  a  W  K  ;  &    �  �  �  �  \  6          	  /  Z  �  �  �  �  �  �  �  �  `    �  �  K     �  �  �    �  �  �  �  �  �  �  �  �  �  �  n  P  -    �  �  \    �  %  1  C  V  ]  O  3        �  �  �  �  Y  )  �  �  R  v            �  �  �  �  m  B    �  �  [    �  &  �  b    �  �  �  �  �  �  ~  N    �  �  7  �  r    x  �  l  �  �  �  �  �  �  y  \  >    �  �  �  �  T  %    �  �  �  {  V  c  u  �  �  �  �  t  d  P  8    �  �  �  K    �  n    �  �  i  R  ;  %    �  �  �  �  �  �  �  �  �  �  �  d  0  �  �  �  �  �  �  �  �  �  j  S  :    �  �  �  �  K     �  �  �    M  b  r  v  �  �  �  �  u  S  $  �  y  �  n  �  �  }  �  �  �  �  }  l  U  7    �  �  �  u  Q  $  �  �    �  A  R  ^  k  s  t  l  Z  G  2      �  �  �  �  B  �  �  %  $  7  u  �  �  �  2  I  ]  c  R  :    �  �  �  D  �  B  �       �  �  �  �  |  ^  ?    �  �  �  �  ^  9    �  �    T  C  4  %      �  �  �  �  �  �  �    P    �  z  �  W  |  �  �  �  �  �  t  Q  )  �  �  �  L    �  �  +  �  �   �  8  0  )  "      
    �  �  �  �  �  �  �  f  K  /     �  y  �  �  �  �  �  �  �  �  �  �  i  @    �  �  o    �  �  [  Q  G  =  2  &      �  �  �  �  �  �  i  O  5        �  4  (        �  �  �  �  �  �  �  �  �  �  �  g  6     �  �  �  �  �  �  �  �  o  Y  D  (    �  �  �    \  7     �  �  �  y  X  :      �      �  �  R    �  i    �  =  �  G  P  :      �  �  �  S  %  �  �  �  l  >    �  �  G  �    	  �  �  �  �  �  �  �  m  C    �  �  ]    �  �  �  �  �  �  �  �  |  R  )  �  �  �  �  \  6    �  �  5  �  _   �  v  �  �  �  �  �  �  �  �  |  o  X  6    �  �  -  �      r  r  |  z  r  _  ;    �  �  &  �  T  �  u  �  ]  y  a   �      �  �  �  �  �  �  �  �  �    X  @  E  J  J  2      �  �  �  �  �  �  `  ;    �  �  {  M    �  M  �  )  J  0  �  �  �    {  w  s  k  b  X  N  E  ;  7  ?  F  M  U  \  d  /  #         �  �  �  �  �  �  �  �  �    p  _  O  >  -  �  �  �  �  �  t  `  I  1    �  �  �  _    �  w  #  �  �  4  2  0  .  ,  )  &  #            �  �  �  �  �  g  M  ]  U  M  E  <  4  ,  %        	     �   �   �   �   �   �   �  �  �  �  �  �  �  �  z  k  X  E  1      �  �  �  �  �  �  �     �  �  �  v  c  O  9  #    �  �  �  �  �  g  H  (    �  �  �  �  �  �  |  [  6    �  �  b    �  1  �  �  �  ~  �  �  �  �  �  �  �  �  �  �  �  �    e  E    �  �  o    �  �  �  �  �  �  r  [  B  '    �  �  �  �  k  ?    �  �  g  _  X  P  G  =  2  '      �  �  �  �  �  }  S    �  *  �  �  �                �  �  �  �  _  4    �  �  ~  �    %  3  2  &    �  �  �  t  J     �  L  �  z  �  !  -  t  n  �  o  T  :  '    �  �  �  x  7  �  �  6  �  F  �  J  
  �  �  �  �  �  �  t  W  8    �  �  �  �  u  G    �  �  O  ?  (  
  �  �  �  Z    �  v    �  <  �  �  V  �  �  b  �  u  �  �  �  �  �  ~  J    �  �  �  �  X    �  X  �      
    �  �  �  �  �  �  �  �  �  �  {  g  S  =  (     �  �  �  �  �  �  �  �  �  �  �  �  �  w  f  U  D  3  "       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  F  �  �          �  �  w  +  
�  
o  	�  	3  :    �  �  �  �  �  h  7  
  �  �  �  }  i  P  ,  �  �  K    �  �  =  �  �  �  z  f  Q  =  (      �  �  �  �  c    �  �  N     �  	�  	�  	v  	c  	K  	%  �  �  �  I    �  f  �  |  �      �  �