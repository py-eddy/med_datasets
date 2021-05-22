CDF       
      obs    D   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?���vȴ:       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�-   max       P�o�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =t�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���Q�   max       @Fz�G�{     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���
=p�    max       @vffffff     
�  +|   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @Q            �  6   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�-        max       @��`           6�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��F   max       =+       7�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B0��       8�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�wI   max       B0z       9�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�Y�   max       C�y       :�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C�       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          i       =   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          U       >   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          C       ?$   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�-   max       P�Z�       @4   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��+I�   max       ?֞�u       AD   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       =t�       BT   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���Q�   max       @Fz�G�{     
�  Cd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��G�z�    max       @vffffff     
�  N   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q            �  X�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�-        max       @�`�           Y,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�       Z<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n   max       ?֜w�kP�     @  [L      h   
      K      !                  /      3   	   $      	   ;      ;   "   d         	   $         7   6            '   >      (   !         
         '                        	   D                  !   "      	   *   /      
N�-P��?N�-ND�P�o�NԢxO�s�N���Nc�N1�GN��7O���PÊO�"O�8#OTr}P*%$OJ9ORİPB�O�SPT=LO�P)�O�� O�P�N��P���O���OR��P۴PP��N�͈N�;�N}ޫOӫ�O��sN�FtO�@O��N���OS_N���N��vO�G�O�xO��"O^�N��CNĈ|O>OI^�N�*mN]�ePk;\NQF�NBg�N@��O��N)OO��O޳�N^PFN�5eO��kOmYcN�5N��=t�<���<���<#�
<o;�`B;o�o��o�o�t��49X�D���D���D���e`B��o��o��C���C���C���t����㼣�
��1��1��1���ͼ��ͼ�����/��/��/��`B��h�����������+�C��C��C����#�
�''49X�49X�<j�@��D���H�9�H�9�H�9�L�ͽL�ͽY��aG��e`B�e`B�m�h��o��o��o���������������������������5NUpz|ypg[5 �����������������������:BOPYYOBB8::::::::::����#i�����{;0
������
#"
����������"���
/DRTH</";<HMUaea_WULH<;:;;;;_acnz����zna________�������������������������������������������
#/682/#
�����KYmz����������zaH>BK������������������������)+-4;;5)����46BO[ht{����th[OA634BN\t���������gO><>9B���������������������
)/2:;?BAC@;"	��CRy�������naZTH;314C�����������������������������������!'&"���������������������||������������w�����������������ww������
 !
�������4:BN[gt��������t[E64lp{�������������{pkl/48<CHUZ\[\XHC</,,,/'*5[fknkjf[NB5' 'BHUaz���������naUGAB��������������������������������������������������������������������������������
�����������������������������*6CP\u����uhO
��������������������mntz���������zwnmmmm��������������������@BDNO[hihe][TODB@@@@BBO[agb[ZOHB;7BBBBBB#0IYbfd_SI<0#��������������������������������������������

���������./1479<?CHJLIHF</...*0<HILLI<0.#TTalmwwz��zvmaYTSRRTmnz����������~zonlkm�����������
&)))








-5Sgt��������tgNB8.-���������������������������������mnqz����znmmmmmmmmmm���#,8;80,
����������������������������
���������)58522)����u{������{spquuuuuuuu���
#/:/(#
����������������������������������������������

������������#&/12/#�g�]�_�g�r�s�y�������s�h�g�g�g�g�g�g�g�g���s�Z�F�7�7�A�Z�s�����������������������������������Ľнսݽ޽�ݽнĽ��������������������ĿǿĿĿ�����������������������	�����|�l�A�2�Z�s��������#��"�8�8������������������üü�������������������ā�t�n�h�[�B�6�)�����!�)�6�B�h�z�{ā�������#�)�/�<�G�<�9�3�/�#�����U�K�H�F�B�B�H�U�Y�\�Z�V�U�U�U�U�U�U�U�U�H�H�H�I�U�\�a�g�n�t�p�n�f�a�U�L�H�H�H�H����������������������������������������ÇÀ�z�x�zÇË×àìùý����������ìàÇ��ƧƔƐƙƖƗƒƚƧ������������������������������$�0�7�8�3�5�.�0�0�$�àÓÇ��|��~ÂÇàù����������üùìà���������������������
��������
�������������þʾ׾���	��"���	���㾥������������������������$�(�&��������	���������������������������������	��	����߿ѿݿ���5�A�Z�g�s�������s�N�A��ּռʼȼ����������ʼּݼ�������ּ��r�t������������!�/�9�<�/�����ּ������ֺԺҺֺ��������������������������ýÿ������,�B�K�Y�X�L�6�������׾ƾľʾ׾����	��.�5�8�:�.�"���T�K�K�D�C�G�T�a�m�z�����������������z�T�ɺȺ����������������ɺκϺҺպʺɺɺɺ���������������������5�B�U�Q�T�f�a�B��꿒�����k�d�_�_�m�y���������ſʿſ���������������������������������!����������������������6�O�k�u�u�i�O�C�6�*������������������������>�'����Ϲ����3�'������
����'�3�5�=�@�A�@�7�3ìâàÞßàìù����������ùìììììì��������������� �������������������������/�(�#�#�)�1�;�H�T�a�m���������z�m�T�H�/D�D�D�D�D�D�D�D�D�D�EEEEEED�D�D�DӾ�����#�(�4�A�M�R�V�M�H�A�4�(�������������)�3�;�G�T�\�^�U�;�.�"�	�������ºֺ�����!�,�)�!������ֺɺ����H�G�;�4�/�*�/�9�;�H�M�T�U�U�T�I�H�H�H�H�[�U�O�L�O�R�[�h�tāĉČčďčā�t�h�[�[�ܹٹϹù¹��¹ùƹϹܹ�����ܹܹܹ����������������������������������������������!�(�-�:�F�l�y�|�w�l�_�S�F�:�-����	���8�=�6�@�Y�f�r��������Y�@�4��ݿѿͿʿ̿п˿ѿ����������������=�;�=�E�T�V�b�o�{ǃǈǉǈ�|�{�o�b�V�I�=D�D�D�D�D�D�D|D�D�D�D�D�D�D�D�D�D�D�D�D���������������������������������������
��#�0�2�0�)�)�#���
�����s�r�\�T�Z�`�s�w�����������������������s�M�J�M�Q�Y�_�f�r�������t�r�f�Y�M�M�M�M¿¹¼¿����������������¿¿¿¿¿¿¿¿�
���´ª¥²���
��/�H�V�S�V�R�J�<�#�
ŭŢŠŔňŔŕŠŪŭŴŹŹŹŭŭŭŭŭŭ�I�C�I�V�Z�b�m�o�x�o�b�V�I�I�I�I�I�I�I�I�Ŀ����������ĿϿɿſĿĿĿĿĿĿĿĿĿĻ���ܻû����ûлܻ����������������������������� ��������������������D�4�*����.�:�G�S�y���������y�k�`�S�D����н��������Ľн����2�<�A�6�����������'�4�5�>�4�'�����������ھ׾̾ʾʾȾɾʾ;׾پ��������^�T�O�N�X�_�l�x�������������������x�l�^�лǻлѻܻ�����'�4�6�9�/�'��	�����EuEpEiEqEuE�E�E�E�E�EuEuEuEuEuEuEuEuEuEuE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� Z    G u 0 H @ v � O ' O K P Z @ = | o j @ ? 9 _ ) Z H 6 T G D J   H R 4 ? g l 9 , M / # @ H N p m N ' ^ H 4 V V E F Z = T > | 6 ^ - S  7  	  �  _  	�  �  �  �  �  �  �  L  �  ^  Y  �    �  �    l  �  ;  �  �  *  )  ^  �  �  �  �     �  �          �  �  +  �  �  �  -  �  G  �  5  X  �  �  �  �  o  l  s  :  o  �     q    5  ,  5  O=+���w<�o;ě���+�D������`B�49X�u��o�\)�q����w�}󶼼j�T����/���ͽ����w����Y���F�49X�H�9��h�u�P�`�,1����T�#�
�'�P��C���j��P��\)��o���}�0 Ž8Q콅�������+�q���q���P�`�q����+�q���ixս�]/�e`B�ixս���u��9X��Q콁%�����
=��G���;d��`BB5�BrB*0�B|�B%{}B$GB�3Bu�BfB"�B+B��A���B9�B3�BZ"B	r�B�(A���A��BBFBB,�'B?:B�B�NB��B#جB
A;B*-B&�B��B��BC�B��BnWBûB��B! �B0��BncB�BMIB�B��B&(B C�B�B�B�%B%ǥA��6B�BB�DBQB	�]B
�_B
�gBfB$�cB�Bc{Bu�B(�6B��B<?B��B��B�DBD�B��B*5BD�B&�B$P�B׆B�/B8�B!��B.�B;�A�~uB>*B>�B)+B	��B?�A�wIA���B>�B,�B@B��B��B?�B#GB	�B)��B?FB��B�dBA�B��B@�B��B�!B!=�B0zB��B��BBsB ~B�*B&D�B c�B6�BAB��B%��A��MB�4B��B��B	�(B:#B
��B�zB#�	B��B	BB>�B(�LB��BA�B�B��B�tA� �A�SiA&�9Av��A���@�F�AؔCA���A�V|Aű<A�d�A�"BB�B	bDA˒�A��`AV�A�o�A�JzA�3Ak�A[x@PjBA���AY�QA�b�@0�A�nAp�A��dA���>�S�?���A��A�,A��C�9�A8fAa��@H+�A��A�/>�Y�A��@�y�@ԪIA��BZkC���A0��A耍A�v-@ݱ�A��9A�7_A���B�AxF�@�V.A�PAGbA0zH@�w�ASf\@�V�@�[�C��C�yA�k"A�v7A&��Av��A��@��8A�tZA�##A�	AŀiA���A�~�B��B	ANA��A���AX�A���A��nA��ZA �A��@LYA���A\�A��(@3��A��Ap��A��B 7/>�dg?��A�w�A�|A��jC�7�A8�xA\�@C�]A�9�Aیj>���A� �@{��@�.zAKBfC��iA2KA��A���@��A�q�A���A�~�BAAw
|@��A��AвA1��@���AS20@��@¶EC��HC�      i   
      K      "                  0      3   
   %      	   <      ;   #   d         	   $         8   7         	   '   ?      )   "         
         (                        	   E                  !   #      
   *   /      
      3         U      "                  '      #      )         3      7      )      %      7   #      '   3            #         )   #               !   '                           5                  %   '                        !         C                                                   1                  7   #         1                     )                  !                              +                                       N�-O�mWN���ND�P�Z�NԢxO��pNzP�N<�N1�GN��7O���O�(�O�;cO7AOTr}OyhO";�ORİOW�~OIPH�_N�,Oo��OmeO���N��P���O���OR��O]}�PClMN�͈N��N}ޫOwoqON�oN�FtO�@O��N���N�*uNU�LN�B�O��O���O��"N�F�N��CNĈ|N�C�O+��N�*mN]�eP*w`NQF�NBg�N@��O���N)OO�H�O?�N^PFNs��O2C7O2�N�5N��  �  	  i  �  9  R  �  �  3  �  E  �  �    z    1  �  �  6  %  3  �  �  �  �  �  ?    A  �  �  �  b  �  �  �  L  g  �  B  H  �  \  �  �  ,  �    1  �       �  �  �  �  �  S  �  ]  �    �  <  �  K  7=t���C�<ě�<#�
�ě�;�`B%   �D�����
�o�t��D����h�T���\)�e`B�o��C���C��<j��t����
�ě��u�ě���/��1���ͼ��ͼ����P�`��h��/��h��h�#�
��w�����+�+�,1�\)�\)�#�
�L�ͽ',1�49X�49X�D���H�9�D���H�9��%�H�9�L�ͽL�ͽ]/�aG��m�h��\)�m�h������������������������������������)5BSadb[N5) ��������������������:BOPYYOBB8::::::::::���#<b{����{I0������
#"
�����������
#/APRHF<# ��;<>HJUaba]WUOH=<;;;;aahnz����znaaaaaaaaa�������������������������������������������#(/572#
������S\amz�������zmaSLMS������������������������������46BO[ht{����th[OA634MNR[gt�������tg[VONM���������������������
)/2:;?BAC@;"	��TVZamz�������{zmaWTT�����������������������������������%$!�������������������������������������������������������������
 !
�������4:BN[gt��������t[E64lp{�������������{pkl/48<CHUZ\[\XHC</,,,/()/59BNVZ\\[VNB54-)(DHUaz���������naUGBD������������������������������������������������������������������� ����������������
����������������������������*6CP\u����uhO
��������������������mntz���������zwnmmmm��������������������NO[hhhd[[POLDENNNNNN9BDO[`ea[XOJB?999999#0IWdb]UPI<0#������������������������������������������� 


 ���������./1479<?CHJLIHF</...*0<HILLI<0.#STXahmuuvqma^TTSSSSSlnz������������zrnml�����������
&)))








>K\gt���������gNB;8>���������������������������������mnqz����znmmmmmmmmmm���#+6950*
�������������������������������������'),*(( ���u{������{spquuuuuuuu�� 
#*%#
�������������������������������������������������

������������#&/12/#�g�]�_�g�r�s�y�������s�h�g�g�g�g�g�g�g�g�s�g�^�X�X�Z�g�������������������������s�������������Ľнӽܽݽ�ݽнĽ��������������������ĿǿĿĿ����������������������	�������w�h�Y�U�X�s����������� ��	�����������������üü������������������������"�)�6�B�`�h�y�z�t�j�[�O�B�6�)��#������"�#�#�/�<�D�<�7�/�)�#�#�#�#�U�S�H�G�C�C�H�U�W�[�Y�U�U�U�U�U�U�U�U�U�H�H�H�I�U�\�a�g�n�t�p�n�f�a�U�L�H�H�H�H����������������������������������������ÇÂ�z�zÇÍÓÙàìù������������ìàÇ��ƳƬƢơƤƧƳ��������������������������
�����������$�0�6�8�3�5�0�-�0�$��ÓÏÈÇÆÇÍÓÜàìùûüùôìèàÓ���������������������
��������
���׾;ʾȾȾоӾ׾����	�����	���������������������������"�&�$������	���������������������������������	��	�5�(�����������(�5�7�A�C�J�K�A�7�5�ּռʼɼ����������ʼּܼ�������ּ������z������������!�,�7�9�5�-���ּ����ֺֺպֺ�����������������������������������)�9�B�F�A�6�)�������׾ξɾ׾����	��&�.�2�5�4�.��T�O�J�J�S�c�m�z�����������������z�m�a�T�ɺȺ����������������ɺκϺҺպʺɺɺɺ���������������������5�B�U�Q�T�f�a�B��꿒�����k�d�_�_�m�y���������ſʿſ���������������������������������!��������*����������*�6�C�O�Z�[�Q�O�C�6�*��������������������;�'����Ϲ����3�'������
����'�3�5�=�@�A�@�7�3ìãàÞàáìù����������ùìììììì��������������� �������������������������;�4�2�4�;�B�H�T�a�f�m�z���������z�T�H�;D�D�D�D�D�D�D�D�D�D�EEEEEE
ED�D�D߾�����#�(�4�A�M�R�V�M�H�A�4�(�������������)�3�;�G�T�\�^�U�;�.�"�	���������ĺֺ�������!���
����ֺɺ��H�G�;�4�/�*�/�9�;�H�M�T�U�U�T�I�H�H�H�H�h�_�[�S�[�[�h�tāăĆĂā�t�h�h�h�h�h�h�ù¹��ùùùϹܹܹ���ܹϹùùùùù������������������������������������������*�!����!�-�:�F�_�l�u�y�s�h�_�M�F�:�*������2�@�M�Y�f����r�m�Y�M�@�4�'��ݿѿͿʿ̿п˿ѿ����������������I�>�F�I�U�V�b�o�{ǂǈǉǈ�{�z�o�b�V�I�ID�D�D�D�D�D�D|D�D�D�D�D�D�D�D�D�D�D�D�D�����������������������������������������
��#�$�%�#���
�����������s�^�V�Z�c�s�~�������������������������M�J�M�Q�Y�_�f�r�������t�r�f�Y�M�M�M�M¿¹¼¿����������������¿¿¿¿¿¿¿¿�
������¹´¿�����
��/�J�H�M�F�;�/�#�
ŭŢŠŔňŔŕŠŪŭŴŹŹŹŭŭŭŭŭŭ�I�C�I�V�Z�b�m�o�x�o�b�V�I�I�I�I�I�I�I�I�Ŀ����������ĿϿɿſĿĿĿĿĿĿĿĿĿĻ���ܻû����ûлܻ���������������������������� ��������������������E�5�+�!����.�:�G�S�p���}�y�q�i�`�S�E����ݽнνнݽ������(�+�2�(�����������'�4�5�>�4�'�����������ܾ׾;̾ʾʾʾ׾ؾ����������l�_�[�V�V�_�e�l�x�~�����������������x�l������������'�-�2�)�'���������EuEpEiEqEuE�E�E�E�E�EuEuEuEuEuEuEuEuEuEuE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� Z   ! G n 0 C D | � O ) , K / Z , 2 | B j < ? P c ! Z H 6 T 5 A J  H ; $ ? g h 9 # \ 0  3 H K p m / " ^ H 1 V V E G Z 6 G > m  O - S  7  �  �  _    �  y  �  �  �  �  1  (  ?  '  �  �  k  �  �  `  �    �  0  {  )  ^  �  �  �  �     �  �    �      �  �  �  w  �  ^  O  �    �  5  �  p  �  �  �  o  l  s  (  o  b  �  q  �  r  N  5  O  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  �  �  �  �  �  �  �  �  �  �  �  �  �  v  i  \  O  B  5  (    �  �    t  �  �  	  	  	  �  �  h  #  �  C  �  �  #  �  f  g  h  h  h  e  a  Z  Q  F  ;  +      �  �  �  �  �  n  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  [  B  )    ]  �    8  7    �  �  o    �  W  �  �  ~  ;  �  �  5  T  R  N  I  C  ;  .       �  �  �  �  �  l  >    �  �  �  �  �  �  �  �  �  z  b  P  t  �  h  ;    �  Z  �  �  5  �  }  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  n  f  ]  U  /  1  2  2  .  *  &  "          	            	  
  �  t  [  @    �  �  �  �  �  q  U  9    �  �  �  �  �  f  E  4  #      �  �  �  �  �  �  �  �  �  �  s  e  6  �  �  �  �  �  �    f  I  %  �  �  �  �  �  O    �  i  F     �  }  �  �  E  l  �  �  �  �  l  >  �  �  \     �  =  �  <  b      �  �  �  �  �  k  Z  Q    �  �  ]    �  M  �  '  �  O  z  �  �  �  >  O  ]  m  x  n  O    �  �  0  �  �  h  �           �  �  �  �  �  �  �  �  �  �  �  r  O  1  9  A  �  �        )  +  +  /  -      
      �  d  �  �   �  �  �  �  �  �  �  �  �  �  o  T  =  &  ,  6  A  L  V  >    �  �  �  �  �  �  �  �  �  �  s  \  C  "    �  �  �  h  B  �  �  !  -  6  �  �    )  6  5  "  �  �  ]  �  V  �  �  �  	  #    �  �  �  t  D    �  �  {  d  7  �  �  ^  �    Y  )  2  $    �  �  �  �  P    �  �  r  0  �  �  *  �     �  �  �  �  �  �  �  �  �  j  I    �  m    �    Y  �  	  h  
�  V  �    u  3  �  �  �  �  �  O    �  �  �  9  
d  	c  �  �  �  �  �  �  �  �  �  }  _  7    �  �  �    �  -  �  "  �  �  �  �  �  �  �  �  �  �  �  m  .  �  �  6  �    �  �  �  �  �  �  �  �  �  �  �  �  �  z  l  n  o  h  ^  T  J  ?  ?  $  �  �  �  �  �  |  U  %  �  �  x  \  (  �  �  1  �  6          �  �  �  �  �  �  h  B    �  �  }  (  �  \   �  A  ;  1  &        �  �  �  �  }  M    �  �  i  #  �  �  �  &  @  V  n  �  �  �  �  �  �  �  m  ,  �  n  �  �  �  Y  �  �  �  �  �  �  k  5    �  �  �  i  &  �  p  �  ?  �   �  �  �    |  �  �  z  c  I  +    �  �  x  F  ,  7    �  �  U  ^  `  \  O  =  '    �  �  �  �  `  1     �  �  Y  
  �  �  �  �  �  �  �  �  �  q  ^  K  5  !    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  L    �  l  �  d  �  �  �  �  �  �  �  �  k  "  
�  
e  	�  	x  �  R  �  �  (  (   �  L  B  7  ,        �  �  �  �  �  �  �  �  y  h  C     �  g  S  E  5  !  	  �  �  �  �  l  c  D  /  �  �  I  �  j  �  �  �  �  �  �  �  b  <    �  �    0  �  �    w  *  g  7  B  ;  3  +  #        �  �  �  �  �  �  �  }  l  [  K  :  �    /  :  C  G  G  =  0    �  �  w  &  �  P  �     �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  S  [  \  \  Y  S  J  ;  *    �  �  �  �  d  =    �  �  �  t  �  �  �  �  �  t  _  F  (    �  �  �  ?  �  �  Q  (  �  �  >  y  z  ~  �  �  }  o  T  ,  �  �  _    �  P  �  �  �  7  ,        �  �  �  �  �  �  �  z  U  *  �  �  �  F  �  h  �  �  �  �  �  �  �  �  }  Q    �  �  [    �  �  0  �  c    �  �  �  �  g  L  /    �  �  �  �  n  S  8    �  �  3  1  /  ,  (        �  �  �  �  m  I  %    �  �  �  �  �  �  �  �  �  �  �  b  <    �  �  �  e  I    �  �  �  S    �  	      �  �  �  }  S  '  �  �  �  n  B    �  �  Y  Q         �  �  �  �  �  �  �  x  ^  C  )    �  �  �  �  /  �  �  �  �  �  y  g  T  A  -      �  �  �  �  �  �  �  �  �  H  |  �  �  �    V    �  �  3  �  �  h  $  �  �    �  �  w  b  L  ;  <  <  =  8  *        �  �  �  �  �  �  �  �  �  }  n  b  W  L  ?  2  %      �  �  �  �  �  �  �  n  �  �  �  s  a  O  =  ,      �  �  �  �  �  �  �  �  I  �  S  S  Q  L  A  .    �  �  �  s  H    �  �  a    �  [  "  �  �  y  o  e  ^  V  O  H  C  =  8  -         �  �  �  �  0  T  [  J    �  �  �  V  $  �  �  �  �  s  8  �  O  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  A  �  �  A    �          
    �  �  �  �  �  �  �  �  �  r  c  V  H  :  �  �  �  �  �  �  �  �  u  _  G  /    �  �  �  �  �  P  �  �  �      /  ;  1    �  �  �  k  '  �  [  �  V  �  �  Z  �  X  �  �  �  �  �  �  j  G    �  �  >  �  �    X  �    K  5    	  �  �  �  �  �  s  [  B  )    �  �  Q    �  �  7  1  +  !    	  �  �  �  �  �  x  Q  %  �  �  }  D    �