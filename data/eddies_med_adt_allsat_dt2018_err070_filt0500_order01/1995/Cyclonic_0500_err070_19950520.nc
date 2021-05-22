CDF       
      obs    H   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?� ě��T        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mʧ�   max       P�~        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       <o        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?fffffg   max       @F@          @  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ٙ����    max       @vk33334     @  ,L   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q�           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @�7@            8   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��F   max       �D��        9<   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�   max       B0"�        :\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�t�   max       B0?�        ;|   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?`�	   max       C�@l        <�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?Pq   max       C�)L        =�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?        >�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A        ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;        A   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mʧ�   max       Pz��        B<   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��*0U3   max       ?��K]�c�        C\   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ě�   max       ;��
        D|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?fffffg   max       @F*=p��
     @  E�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @vk33334     @  P�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q�           �  \   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @�F�            \�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�        ]�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n   max       ?Ͼvȴ9X     �  ^�                                                   <               	                                             	            >               -                           *         ?                     	                     OCq�N�M�Oj�?O�ٜN��JOr�FO�fO
#`N�N)NOJ�P �NHO�E0O��P(?�P�6M�"�N	��OLN/��N�>�O�ܤNb�O(��O�:�N#��O':
O�a^NIOC�N��Nc��N`:O)קN@��O��N���O~�N���P�~O ��N�N��Oh�O��HOr�bO^ޓO�y�M�@6O��N���N��Mʧ�Pz��O8N��O���O}��O�kNW�O��*O�4�O���N�B�O"1O��N���N�:�N��#N�IN�5`<o;�`B;��
�o�o�o�o�o�o�t��t��t��#�
�#�
�49X�49X�49X�D���T���e`B�e`B�e`B�e`B�u�u��C���t����
��1��1��1��1��1��1��9X��9X��j��j��j�ě��ě����ͼ�`B���o�+�C��C��C��\)�t��t���w�#�
�#�
�',1�,1�0 Ž0 Ž0 Ž49X�8Q�<j�H�9�T���T���e`B�y�#�� Ž�E��ě�{�������������|zutu{��������������������{�����������������}{����������������������� ���������������
"
 �����
#/<HLLI</#�� #/<EHUX\_UH<5/&#"  8<=HUX]`]XUH@<:98888��������������������	'+,/0/0*)����������������������$'��-:;@@=6)�����cmz���������ng[WX\c�����������������������������������������������������������468BELOQOFB:62444444����!),/)����NO[chhh`[WOENNNNNNNNW[cht}�������thh[VWW�� 7CO\b^^XR@6*����������������������)6BIOQQOOJB61)������)1)�����fht~ztlhgbfffffffffftz��������������zyztFTamwwuuz{zpmaTNHDDF9BNPRPNKB>9999999999MNP[]jy������tg[UOLMFHRU]abaUPHEFFFFFFFF����������������������������������������eghltz���������togee#*/20/$#!�������������������������������������������������������������������������������� 37-���������� ���������������������������������������������=BBN[]gjnpmg[NHB;:==nv����������������onLN[gt��������tg[MKJL��������������������'0:Uen{������{bU<#'��������������������������������������������������������������zz����~zvxzzzzzzzzzz�����)6B9)������NOQdhjtw}ztoig[ZPON##*/<A<61/$#"!######!)6BP\dfeb[OB6)%)+5BDN[`gjqokf[NB8*)UUZainz�����zona\UUqt��������~trqqqqqqYgt�����������rg[URYZgt�����������xqh[SZ�������	�����������������������������������3Dbn��������{nbUB==3$%$

9<IMQTTSLI<;75579999���������������uz}�����������xuz|wu������	��������ѻ����������������ûлܻ���������ܻл��������������}������������������������������������������������������	��������z�q�u�z�������������������������������z¦±²µ·²¯¦ÞÓÇÂÇÑàìù���������������ùìÞ��������$�$�)�B�O�h�c�W�V�O�6�)��������������������������������������������
��� �
��#�/�<�D�B�<�5�/�#�����b�W�U�R�U�b�g�n�u�t�n�h�b�b�b�b�b�b�b�b�����������!�-�:�F�G�H�=�:�-�!���A�>�?�E�D�R�f������ҾҾϾɾ�������f�A�ĿĿ¿��ĿѿԿֿҿѿĿĿĿĿĿĿĿĿĿľM�A�8�-�'�4�M�Z�s�������������s�f�Z�M��� ��)�:�B�O�hĈĚġčā�t�h�[�O�6�ĮĦĳ�����������
�0�<�I�Q�I�0�����ĿĮ�#��$�+�4�@�Y�f����������������}�f�Y�#�A�<�4�(����(�-�4�?�A�C�C�A�A�A�A�A�A�S�Q�S�[�_�l�s�x�{�x�w�l�f�_�S�S�S�S�S�S����������������������������������������ÓÒÊÓÞàãìïìéàÓÓÓÓÓÓÓÓ����������������������
���
�
�������;�.�"����ʾ��������ʾ���	�"�*�2�E�@�;Ç�~�z�r�u�zÇÏÓÒÇÇÇÇÇÇÇÇÇÇ���������������Ŀѿݿ޿������ٿѿĿ��0�$������$�0�=�I�N�K�E�D�B�>�?�=�0�e�[�Y�e�r�{�~��~�r�e�e�e�e�e�e�e�e�e�e������������������"�"�(�(����ƪƤƢƧƳ�����������������������������������������������������������������ѿǿĿ��Ŀѿݿ�������������ݿ��g�e�Z�V�Z�d�g�i�r�s�w�s�g�g�g�g�g�g�g�g�g�_�Z�Q�Z�_�g�r�s�������x�s�g�g�g�g�g�g�H�A�<�9�<�H�U�a�a�a�e�a�]�U�H�H�H�H�H�H¿µ²¦¦²¿����������������¿����������������������������6�3�*��*�6�C�O�\�b�h�u�zƀ�u�`�\�O�C�6��	�������(�.�0�)�(�������àÙÓÐÌÇÄÉÓìù����������ûùìà�5�4�0�5�9�A�K�N�Y�Z�]�\�Z�N�C�A�5�5�5�5�������r�o�z��������!�.�9�8�+� ���ʼ���ŠŚŚŠŦŭųŹž����������������ŭŪŠā��t�p�t�wāċčĐčĉāāāāāāāā�������v�m�h�_�`�m�y�����������������������������������������������������������ѻ����{�m�j�c�`�g�d�l�v��������������������������������	���#�)�'�"��	�����پ־׾ھ����	��$�(�"� ���	����������s�W�N�J�I�J�Z�g�s�������������������ʾǾ������ʾ̾׾����׾ʾʾʾʾʾʾʾ���������5�B�N�T�[�`�^�N�H�B�)���#������#�0�<�D�D�<�3�0�#�#�#�#�#�#�ѿʿǿɿѿտݿ����������ݿԿѿѿѿ�������������	��	����������������������`�0������ �:�M�v�����Ľн��������y�`���������}�������������Žнݽݽӽн�����E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������s�h�c�g�s���������������������������*�������������*�6�C�I�L�I�C�6�*�z�y�m�e�a�^�\�\�a�m�z�}�������������z�z��������'�3�7�6�3�+�'������ŔŋňŋŌőŠŭŲŹ��������������ŭŠŔĳįĭĨĦĳĿ�������������������Ŀĳ�r�Y�L�C�;�7�@�L�Y�e�r�~�������������~�r��ý������������������������������������Ç�|�z�n�zÇÓàìù����������ùìàÓÇ�ܻû����ûл�����'�0�1�(�.�-�������6�2�/�6�B�O�[�h�q�h�e�[�O�B�6�6�6�6�6�6����������(�4�8�6�4�(������������<�8�/�)�(�/�<�H�U�a�h�a�]�U�H�>�<�<�<�<�ܹعܹ�������'�/�'�$���������ܼM�K�@�@�=�;�5�@�M�V�Y�f�g�k�l�f�c�Y�M�M J T K F K a V T 6 X = @ g ) X N = w u > 8 S � @ 7 x Q J L M S T 0 F D 5 �  1 B ` Y ' T ( _ F  f � F $ X ` R = {    D 9 ? < ) k z ? ^ T G o E  �  @    q  ,  N  ?  O    b  �  �  b  �  -  G    `  s  �  <  "  �  �  t  �  (  �  G  D  �  )  ]  �  �  \  d  �  �  �  �  T  2  P  M  �    �  ?  Y  >  �  �    7  x  r  Y  �  :  n  !  r  X  �  �  -  �    �  8  0����D���ě���/���㼣�
�t���9X����49X���t��D����P��w�,1��\)�e`B��C����ͼ�C���j�t���9X��9X��`B�ě����#�
�����t���j��h��h��P�ě�����`B�]/����{�o�o��P�<j���-�q���e`B�H�9��w�m�h�'Y��,1���T�H�9�D�������}�]/�H�9�aG��y�#���P�m�h��%��\)��7L���-����������FB��B^&BQ&B}B
'B��B��B�}Bk�B B��B �@BޱBj�A��`B��B!�(B(\B=EBoMB$�B>�B0"�BX�B��B��BSB �3A�B��B	G3B�B��B!��B
	�B��B ��B��B�]B�!B,��B0?BD[B*�B��B��B	��B#MB'�B��B��B�_B�&A�ܮB��B��BS�BB,�B�B�cB
!�B
@�B"��BcEB7B'�B�B&�$B&B�SB�gB�B�=B��BC�B@dB��B�B��BC�BABIB!?�B�BQ�A��lB�B!@ B�+B@B?|B?�BF�B0?�BF�B�B�XB<�B+�A�t�B�IB	RoB��B�B!��B	�kB��B �B�[B�B�'B-��BA�BIB*�XB�B��B	� B>�B(��B?B��B,sB��A��B�KB�>B@�B��B؄B��B�<B
 ]B
��B"��B��B8�B'��B*�B&�WB
�B��B��@��A�<YA�ȁA��A��nA��A�J�A�ZA�g�A��h@g��AF�Ay�AAVA�jA��@�5A8�#@�W�A�ƗA�T�A��!AY|A�M^Az��B
��?��'A���B�'Ar��AV�A��JA��yA�*+A���A�8BFNA4�1A�)�A��"A�A�{A݈�Ao0�A��i@���AZt�AY-�A�AR��A��A�A/A}\IA���A�A#��C�@lA�`JA�LA��?��A�_lA�R�?�*�A�v�A�S`@��A��>A3�NA�Q%?`�	@�1�@���A�b�A��6A���A��!A�AA�g�A���A�o�A� �@d}�AI�Az�AA<\A�|�A��@���A9@���A�LIAʁ�A��{AY��A�D�Ax��B
8�?��A�sB�JAqA�PA�e2A���AŁ<A���A��!B �cA4�zA��A���A�^A��0A�}�Ao�A��&@��=AZ��AY�A�u}AS(qA��}A�&A})A�w�AŀA!fC�)LA�e�A��A�!�?�3�A���A��?��bAϊ�A˛�@�DA٬gA5	sAÏ_?Pq@���                                                    <               
                                 	   	         	             >               -                           *   	      ?                     
                                                         '      #   %   /   )                  +         !                                             A               !         #                  ;                                    '                                                   %            /   %                           !                                             ;                        !                  ;                                                   O�_N�ՂOEl9O8�vN��JOr�FN�[O
#`N�W�N)NOJ�P�zNHO�y�O�D;P(?�O�hM�"�N	��OLN/��N���O*(rNb�O(��O�:�N#��N�O�a^NIOC�N��Nc��N`:O)קN@��O��N���O��N���Pm��O ��N�Nl#Oh�O,V�O:�dOQ��O�SFM�@6O��N���N��BMʧ�Pz��O8N��O��OR�yN轅NW�O��*O�4�O}��N�B�O��O���N���N�:�N��#N�#DN�5`  c  P  �  �  �  �  [  �  Y  `  N  �  �  e  L  �  �  :  m  �  �  �  U    �  V  �  }    �  �  �  �  �  Z  �  �    �  �  �  3  �  @      �  �  �  b  (  P  R  �    I  .  	�  5  n  �  D  W  �  �  �  �      a  ]  ;D��;��
;�o�t��o�o��j�o�#�
�t��t��#�
�#�
��o�e`B�49X��C��D���T���e`B�e`B�u��j�u�u��C���t���9X��1��1��1��1��1��1��9X��9X��j��j�+�ě���/���ͼ�`B�+�o�8Q���\)�\)�\)�t��t��#�
�#�
�#�
�',1�H�9�8Q�49X�0 Ž49X�8Q�L�ͽH�9�Y��ixսixսy�#�� Ž�^5�ě�xz{������������zxwx��������������������~������������������~�������������������������� ���������������
"
 ����#,/<A<;/# #/<EHUX\_UH<5/&#"  :<@HUU[][VUHF=<;::::��������������������	'+,/0/0*)����������������������$'��).49:5) ����^emz���������zsj]YY^�����������������������������������������������������������468BELOQOFB:62444444����!),/)����NO[chhh`[WOENNNNNNNNY[ehtx�������tmh[XYY*069COPQPMHC6*��������������������)6BIOQQOOJB61)������)1)�����fht~ztlhgbffffffffff�����������������~��FTamwwuuz{zpmaTNHDDF9BNPRPNKB>9999999999MNP[]jy������tg[UOLMFHRU]abaUPHEFFFFFFFF����������������������������������������eghltz���������togee#*/20/$#!����������������������������������������������������������������������������������15*���������� ���������������������������������������������=BBN[]gjnpmg[NHB;:==�����������������~z�NNU[gt������{tgb[QNN��������������������-0=Uhn{�����{bUI<0*-��������������������������������������������������������������zz����~zvxzzzzzzzzzz�����)6B9)������NOQdhjtw}ztoig[ZPON##*/<A<61/$#"!######$)6BLX_ba^[OB6)" $-5:BJN[emkga[NB:5,*-[alnz�����}zna\V[[[[qt��������~trqqqqqqYgt�����������rg[URYZgt�����������xqh[SZ��������������������������������������������EK[bkwzxywnibUJFDDAE#$#	9<IMQTTSLI<;75579999���������������vz|������������zzvv������	��������ѻлŻû��������������ûлܻ�������ܻ�����������������������������������������������������������������	����	����������{�z�z�~����������������������������¦±²µ·²¯¦ÞÓÇÂÇÑàìù���������������ùìÞ�)�"�#�)�,�.�6�A�B�J�I�B�=�6�)�)�)�)�)�)�������������������������������������������
���
��#�/�<�@�>�<�/�+�#�����b�W�U�R�U�b�g�n�u�t�n�h�b�b�b�b�b�b�b�b�����������!�-�:�F�G�H�=�:�-�!���f�M�A�G�G�T�d������оѾѾ;Ǿ�������f�ĿĿ¿��ĿѿԿֿҿѿĿĿĿĿĿĿĿĿĿľM�G�B�7�>�A�H�M�Z�s�����������s�f�Z�M�6�����)�6�B�O�h�t�čĆā�t�h�[�O�6ĮĦĳ�����������
�0�<�I�Q�I�0�����ĿĮ�-�'��'�1�@�Y�f��������������v�f�Y�@�-�A�<�4�(����(�-�4�?�A�C�C�A�A�A�A�A�A�S�Q�S�[�_�l�s�x�{�x�w�l�f�_�S�S�S�S�S�S����������������������������������������ÓÒÊÓÞàãìïìéàÓÓÓÓÓÓÓÓ����������������������
���
�� ������	�����ھ׾̾Ӿ׾����	���'�*�"��	Ç�~�z�r�u�zÇÏÓÒÇÇÇÇÇÇÇÇÇÇ���������������Ŀѿݿ޿������ٿѿĿ��0�$������$�0�=�I�N�K�E�D�B�>�?�=�0�e�[�Y�e�r�{�~��~�r�e�e�e�e�e�e�e�e�e�e�������������������������ƪƤƢƧƳ�����������������������������������������������������������������ѿǿĿ��Ŀѿݿ�������������ݿ��g�e�Z�V�Z�d�g�i�r�s�w�s�g�g�g�g�g�g�g�g�g�_�Z�Q�Z�_�g�r�s�������x�s�g�g�g�g�g�g�H�A�<�9�<�H�U�a�a�a�e�a�]�U�H�H�H�H�H�H¿µ²¦¦²¿����������������¿����������������������������6�3�*��*�6�C�O�\�b�h�u�zƀ�u�`�\�O�C�6��	�������(�.�0�)�(�������àÕÓÒÐÑÓàìùù����þùìàààà�5�4�0�5�9�A�K�N�Y�Z�]�\�Z�N�C�A�5�5�5�5���������|����������!�-�7�6�)���ؼʼ�ŠŚŚŠŦŭųŹž����������������ŭŪŠā��t�p�t�wāċčĐčĉāāāāāāāā���}�y�p�s�y�������������������������������������������������������������������ѻ��x�t�o�l�j�i�l�x�������������������������������������	���"�&�$�"���	�����ھ׾׾ܾ����	��"�'�"����	���������s�Y�N�L�J�M�Z�s���������������������ʾǾ������ʾ̾׾����׾ʾʾʾʾʾʾʾ���������5�B�N�T�[�`�^�N�H�B�)���#������#�0�<�D�D�<�3�0�#�#�#�#�#�#�ѿϿȿʿѿֿݿ����������ݿѿѿѿѿ�������������	��	����������������������`�0������ �:�M�v�����Ľн��������y�`���������}�������������Žнݽݽӽн�����E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��s�l�f�j�s�����������������������������s����
������*�6�C�F�J�H�C�?�6�*��m�f�a�`�^�^�a�m�z�{�����������z�m�m�m�m��������'�3�7�6�3�+�'������ŔŋňŋŌőŠŭŲŹ��������������ŭŠŔĳįĭĨĦĳĿ�������������������Ŀĳ�L�F�>�<�B�L�R�Y�e�r�~�����������~�r�Y�L��ý������������������������������������ÓÍÇ��{ÇÓàì÷ú����������ùìàÓ�лû������ûлܻ�����&�'������ܻ��B�6�<�B�O�[�h�o�h�d�[�O�B�B�B�B�B�B�B�B����������(�4�8�6�4�(������������<�8�/�)�(�/�<�H�U�a�h�a�]�U�H�>�<�<�<�<����������� ����'�*�'�!������M�K�@�@�=�;�5�@�M�V�Y�f�g�k�l�f�c�Y�M�M E A G : K a U T 7 X = 8 g / R N 7 w u > 8 U c @ 7 x Q 7 L M S T 0 F D 5 �  9 B ] Y ' < ( A <  \ � F $ X ` R = { !  E 9 ? < ( k | > Q T G ^ E  C  �  �  �  ,  N  �  O  �  b  �  k  b  #  �  G    `  s  �  <    �  �  t  �  (  �  G  D  �  )  ]  �  �  \  d  �  G  �  }  T  2  l  M  |  �  �  �  Y  >  �  �    7  x  r    �    n  !  r  �  �  �    �    �  �  0  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�    R  ^  b  b  Z  K  5    �  �  �  h  8    �  �  ]    �  7  ?  G  L  O  H  >  /      �  �  �  �  v  R  +    m   �  �  �  �  �  �  �  o  V  E  9  5       �  �  �  �  �    X    ?  \  o  �  �  �  �  y  k  Y  B  #  �  �  P  �  |  I  �  �  �  �  �  �  x  g  N  ,  	  �  �  �  c  2  �  �  �  W  !  �  �  y  b  =  !  (  ,  -  !    �  �  �  �  �  j  K  *    �    !  0  3  /  +  "    #  M  Y  Q  E  3    �  �  �  �  �  �  �  �  z  s  l  b  R  ?    �  �  j  $  �  �  A  �  �  O  S  V  X  W  S  I  =  &    �  �  �  n  6  �  �  1   �   b  `  [  V  Q  L  F  A  <  7  2  -  )  %  !            
  N  D  9  -       �  �  �  �  I    �  �  a  %  �  �  {  �  �  �  �  �  �  �  p  [  A     �  �  �  d  3  �  �  t     u  �  �  �  �  �  �  �  �  �  �  �  |  r  h  ]  S  I  >  4  *  J  Q  [  a  d  _  R  @  )    �  �  �  �  n  1  �  �    �  H  =  K  7       �  �  �    J    �  _     �  Y  )  �  6  �  �  �  j  T  ?  &  !       �  �  �  k  E    �  �  `  ?  �  �  �  �  �  x  H    �  l  
  �  A  B  7    �  H  b  S  :  8  6  5  3  1  /  -  +  *  %           �   �   �   �   �  m  p  t  w  z  x  h  X  H  8    �  �  �  |  i  \  P  C  7  �  �  �  �  �  �  �  �  �  �  �  }  m  T  5    �  �  T   �  �  �  �  �  �  �  �  �  v  j  ^  Q  E  ?  D  J  P  U  [  a  �  �  �  �  �  �  �  w  a  H  -    �  �  �  �  `  1   �   �       &  /  2  )     @  S  E  .    �  �  �    K    �  k      �  �  �  �  �  �  u  Z  >  !    �  �  �  �  n  >    �  �  �  �  �  �  �  �  �  �  �  �  t  _  G  ,     �   �   �  V  F  ;  =  5  #         �  �  �  �  e  :    �  �  Y    �  �  �  �  �  �  �  r  b  R  B  2  "    �  �  R  T  V  W  H  b  z  |  }  |  z  t  m  b  W  J  <  -    
  �  �  �  �    
  �  �  �  �  �  b  =    �  �  �  \  "  �  �  T    ^  �  �  �  |  s  h  ]  R  D  2      �  �  �  �  {  Z  9    �  �  �  �  �  �  w  g  X  H  4  #    �  �  �  h  .  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  e  T  D  3  "    �  �  �  �  �  �  �  w  _  F  +    �  �  �  �  _  6  	   �  �  �  �  �  �  �  y  f  P  :  "  
  �  �  �  �  R  &     �  Z  W  Q  G  9  #  �  �  �  �  b  5    �  �  �  r  ]  l  �  �  �  �  |  v  q  k  f  `  [  Q  B  3  $       �   �   �   �  �  �  �  w  _  H  6  $      �  �  �  �  �  �  �  �  w  a    	  �  �  �  �  �  �  �  �  �  �  �  }  p  c  N  7  !    E  m  �  �  �  �  �  �  �  S     �  �  d  (  �  T  �  �   �  �  �  �  �  �  �    p  `  O  ;  %    �  �  �  �  �  j  I  �  �  �  �  �  [  !  �  �  �  �  v  `    �  C  �  �  L  �  3  "      �  �  �  �  �  �  {  f  \  Q  G  ?  7  6  7  8  �  �  �  �  x  m  \  J  9  '    �  �  �  �  p  C     �   �  �      *  0  7  ;  =  @  7  ,  !      �  �  �  �  �  m            �  �  �  �  �  m  F    �  �  �  L    �  _  t  �  �  �        �  �  l     �  �  K  �  W  �    G  p  �  �  �  �  �  �  �  �  �  �  n  J  !  �  �  l  �  n  �    �  �  �  �  �  �  g  D    �  �  �  T  #  �  �  D  �  Y   �  �  �  �  �  �  �  �  e  G  )    �  �  �  �  �  s  A   �   �  b  `  ]  [  X  P  5       �  �  �  �  w  [  >  !     �   �  (  #    �  �  �  �  �  a  5     �  �  A  �  �  �  H  �  �  P  L  G  B  =  5  -  %        �  �  �  �  �  �  �  �  �  J  P  K  <  $  	  �  �  �  �  t  T  /  �  �  �  B  �  v    �  �  �  z  c  M  7  !     �   �   �   �   �   �   �   �   �   �   s      �  �  �  �  �  �  z  K    �  �  �  �  Z  �  #  :  y  I  E  A  =  8  3  *  "      �  �  �  �  �  �  �  �  �  �  .  +  '  $        �  �  �  v  ^  G  1      	  X  �  �  	�  	�  	�  	�  	�  	q  	V  	0  	  �  �  l  %  �  2  �  �  X  �  �  )  /  5  3  2  -  )  #      �  �  �  �  m  8    �  �  �  f  k  n  l  i  c  Y  M  ?  0    	  �  �  �    O    �  �  �  �  �  �  x  n  d  `  _  ^  Z  T  M  7  �  �  �  Z  %  �  D  0      �  �  �  �  �  �  �  x  g  R  =  %    �  �  �  W  S  P  N  M  M  K  F  <  +    �  �  �  �  q  S  2    �  �  �  �  �  �  �  �  �  �  �  r  J    �  �  W    �  �  O  �  �  �  �  �  �  �  �  �  �  �  �  �  .  +  (  "        �  �  �  �  �  �  �  �  f  I  (    �  �  �  �  o  <    �  a  \  b  �  �  �  �  �  �    l  W  @  ,      �  �  z   �  �  �      �  �  �  �  �  |  e  L  2    �  �  �  W    �    �  �  �  �  �  �  p  N  ,  	  �  �  �  {  R  %  �  �  �  a  U  J  9  &    �  �  �  �  �  q  U  ;  "     �  �  h  �  7  Q  [  Y  Q  J  >  *    �  �    F  
  �  �  9  �  �  /       �  �  �  �  �  r  =    �  �  @    �  �  ]  �  K  �