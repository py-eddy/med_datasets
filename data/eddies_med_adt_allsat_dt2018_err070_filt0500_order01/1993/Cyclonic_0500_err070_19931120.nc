CDF       
      obs    M   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���S���     4  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N$��   max       P�ٛ     4  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <�`B     4      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?@        max       @FxQ��       !H   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @vg\(�       -P   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q@           �  9X   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�1@         4  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <���     4  ;(   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B0)     4  <\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B0?/     4  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >���   max       C��     4  >�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >k��   max       C�      4  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          m     4  A,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =     4  B`   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3     4  C�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�%�   max       Pc\,     4  D�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�*�0��   max       ?ӯ���s     4  E�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <�`B     4  G0   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?k��Q�   max       @FC�
=p�       Hd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��p��
>    max       @vg33334       Tl   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q            �  `t   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          4  a   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�     4  bD   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?|�1&�y   max       ?ӯ���s       cx      *            &               %   8   	   &                                    
                     H      #         l                               	      
      &      A   
         2      '            -         +            "   
               N-ԾO��N\��N]��N���O\5�NZ�gN7�,N'C'OF��O�pPr9jO��Pf O	M8O�4N[�O�3�N$��O�c�O��bN�{^Owm�N�2�O�+N(��NCS�N7s�N>y:NI%}OsxNՅ�P@7OX֛O�!N��]N'nP�ٛN��eNY0N�R�O��(N1Q�Od�uN��N�GmN��N���Oi�dN�=~N�ˑO�vOb�'P&�N�<DO��O��DO��yN��N��N�n�N@�!N1��OH�O]R�O<��P�5NM'�N�VOY�Op]�OJNOz��O�O-��N�9N��;<�`B<�/<�1<��
<���<�o<t�;��
;��
��o�ě��t��t��#�
�49X�49X�D���T���e`B��o��o��o��C���t���1��j��j��j��j������`B��`B��`B��h�����o������w��w��w�',1�0 Ž49X�49X�8Q�@��D���D���H�9�L�ͽL�ͽP�`�P�`�T���Y��Y��]/�]/�aG��aG��aG��aG��aG��e`B�m�h�m�h�}󶽅������+��+��C������� )*)#��������������������()+5>BGB5-)"((((((xz��������zxxxxxxxxx������������������������������~|�#)/<A@=<:/)#����������������������

�������������JN[gt����zxtg[SNJEBJ���6;;8) �����8@Tmz������}[THF><58#/7<EHIHC</#Qay����������zaTNNQQHKKIHB</-##%/<HHH�6CMlp���wh[C����

��������������������������������������������������������

���������#/<HWfmaUH</#Q[hkt�������tthe[SQQ�������������������������������������������������������������������������������������������������������������������������������������������������


�����������!)95-)&�����=BCN[grrrg[NKB======y���������������vuy")/<IUVVTTQQHD<)#"��������������������������������)+3)))?Y`^_g�����������t[?��������������������#--/0/&##+/1<DB</-#�������������������������������������{�����������������~{kty������������tkkkk`amtzz|�zymea^\````��������������������NOX[hsonh[ZOKGNNNNNN��������������������./<CGC><3/'',-......BBFOP[hhokheb[OKB>BB��)/2-'��������U]gty���������tgb[WUEKUanx������zn\UIBAE�������������������� %).8IUbnspjaNI<0# t�~}��������������ttpt�������������tlffpst���������}tmjlssss�����


���������BFOU[]``_^][POLB:<?B������������������������������������������

 #$#!
������ #3HUaigaUPH3/%%##  \anz����������zna\Z\��������������������������������������������������������
 #0;EME<70#
��)5BNNB@6)����x{|���������{xtsstzxAEGN[gtvz}�tg[NHB@A��������������������;<IU^hmjiebUKI><656;���������������������������������������޺ֺպԺֺ��������ֺֺֺֺֺֺֺ���
�����������������
��/�2�<�D�B�/�#�ƧƦƧƬƳƿ��������������ƳƧƧƧƧƧƧùòñìèìù����������ùùùùùùùù¦¦²¿��������¿¾²¦�����������ʼּ�������������ּʼ��)�'�)�.�0�6�B�O�R�O�I�B�:�6�)�)�)�)�)�)�U�S�L�K�L�U�a�d�a�]�^�W�U�U�U�U�U�U�U�U�������������������������������������������������������������ĿʿͿʿĿ�����������i�h�l�z�}���������ʼռݼּϼ�������������������(�A�r�����������s�Z�5��"����"�(�/�9�;�<�H�L�P�O�H�D�;�/�"�"č�}čĕĒĈčĚĦ������	��������Ħč��$�0�=�>�I�O�J�I�D�=�0�$��������G�.�"������.�;�S�Z�V�`�e�T�I�I�R�Q�G�M�J�B�M�Y�f�p�r�t�r�f�Y�M�M�M�M�M�M�M�M�ݿѿƿĿ������Ŀѿܿ��� ���������ùõ÷öù��������������ùùùùùùùù����������������������������������������g�Z�S�N�N�S�Z�g�s�~�����������������s�g�/�+�$�#���#�/�<�H�U�W�U�U�J�H�>�<�/�/�m�a�T�H�@�>�@�H�T�m�z�����������������mìæà×ÕÓÐÇÂÇÉÓØàæì��ùôì������r�f�c�b�f�r���������������������ÓÓÏÓàììîìàÓÓÓÓÓÓÓÓÓÓ���������������������������������������������������$�'�$��������������������� ���������������������������������������������y���������������������������������������������������������������������������׿Ŀ����u�l�h�j�t�������Ŀ����(�����ľf�Z�M�D�A�J�M�Z�f�s����������������s�f�n�b�U�J�=�;�B�I�b�n�{ŔŠūůŬŠń�{�n��ŹŹŴŹ�������������������������������I�C�B�H�I�V�b�b�b�Z�V�L�I�I�I�I�I�I�I�I�
��²�[�C�3�8�7�A�[¦¿�������
��
���������������ʾ׾����׾ʾ����������n�c�a�Y�a�n�u�z�}ÇÌÇ�z�u�n�n�n�n�n�nàà×ÓÑÓÜàìïöùýù÷ìàààà���������������ɺֺ���!�5�@�-�!���ֺ�������"�$�+�0�0�0�(�$����������������������(�4�<�A�@�9�4�+�(�������������������������������������������ƾƷ����������������������������������������������������������������������������������������������������������������������������������$�(�&�!���D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��Ϲιù����������ùϹٹܹ����޹ܹϹ�ÝÜâíþ�������������������ùìÝ��������������ŹŹ��������������빝�����~�����Ϲ��@�L�Y�C�'����ܹù����e�_�`�e�l�r�~�������������������~�r�e�e���������s�c�]�X�[�g�s�������������������o�Y�@�3�*�*�.�;�@�J�L�e�m�w������|�t�o�������������������	��#�3�8�9�7�/�"��佅�������������������������������������D�D�D�D�D�D�D�D�D�D�D�EEEE D�D�D�D�Dӿ����|�y�w�y�������������ĿſĿ����������b�^�a�b�o�{ǀ�}�{�o�b�b�b�b�b�b�b�b�b�bǈ�|�{�w�y�{�}ǈǔǘǕǔǈǈǈǈǈǈǈǈEEEEE*E7ECEMEPEWE\E_E\E[EREPECE7E*EEiE_EiEuE~E�E�E�E�E�E�E�E�E�E�E�E�E�EuEi�ʾ¾��������ʾӾ׾ھ������������׾ʼf�`�f�����ּ�����������ʼ����r�f�:�1�:�>�F�S�_�l�q�l�_�_�S�F�:�:�:�:�:�:�x�v�x�|�������������������������������x�Ľ������������������н۽������ݽнĿ"���������	��"�.�5�9�@�>�;�.�"������������'�4�@�I�M�E�@�4�'�����Ěčā�u�}ćĕĚĦĳĿ����������ĿĳĦĚ����������Ŀľ�����������
���������ػ����������ûлܻ�������������ܻлû������������������������������������������(�5�;�;�5�/�(����������� C : F N F $ o � F ' 9 _ 3 2  r : J � $ ? F b � M L _ d Y N f C I 6 7 . h N S E G L w  b T j B / _ 5 2 N ~ S b R R $ < R 4 A L K $ u u Z  Z S 8 8 ! j      L  b  �  w  
  �  �  �  2  �    �  6  �  +    _  D  �  �  B  %  ?    i  4  o  i  I  W  C    �  �  �  �  m    �  e  �  g  �  �  �  �  2  �  �  �  �  A  #  �    �  �    �  $  4  \  b  T  �  �  w  �  1  �  A  Z  �  Y  o  �  <�C��u<���<�C�<t���9X;�o��`B%   ��t��,1��o��C��@���h�#�
���
�'��
�@��,1��/��h���ͽ8Q�+��������h���@��\)���ͽP�`����\)�����,1�8Q�49X��hs�49X��C��L�ͽT���P�`�Y���hs�ixս����-��+���y�#��o���w��񪽅���j�}�q���u���ͽ��t�������%��\)�� Žȴ9���P��vɽ�����w��1��`BB�UB��B�B X%B�XB�BZB��B#�vB��B��A��B�B W�Bu�B0)B$  B�B��B�B��B'�BvB�B &B�B%fB��B�B#�Bn�B��B*��BПBP�BK�BQ�B
�kB )�B�3BB"H2B��Bc\B�MA���B:�BRqB�By�B)LBB	�]B(B�B&�B >GB�BB��B�B�iB�B��B��B�sB.�B,��B�IB&YB%�B,,B)4�B� BX�B'B�B�<B�@B�FB?�B vcB��BN~B@lB%SB#��B�B&�A��B�B %�B�3B0?/B#�.B�1B��B:�B�wBBVB�LB�7BɞB9�B@�B��B��B#��B?�B��B*��B�B�bBl)B�DBA�B 9NB��B��B!��B
�B�0B�A��BB$�B��B��B�	B?�B<�B	�`B@B��B&��B�:B>7B�mB��B�B�B�MB��B�ABs�B,?�BK2BB�B%CB>�B)=�B��B9tB&�kBb�B�x@D("A�TSB��Aͥ�A��A&TA��VAŚ[@��At��@A��A�.�A�0B
$tA`�!@�LNA}�RA��A��qA��nA�a9A�bA�-�@��A˖A��@B޶A�Z�@�הA�vkA�n%AyH�AB�A�O=A�+�B��A�f�APp�A�ҬA�)@GݝB	�A5�@���B��Bb@A��mA��KC���>���A�ԀA�%�?�w@��A��?֣sA��A ��C�*6Ar�B�B��C���C��AS[7@���@�Tg@��A'PA_"V@ɹ�AࢁA�~@�,A�l*A���@D!tA�B@?A���A��UA��A�vYA�z�@��7At��@�)�A��#A��7A�5B
2%A^Ϻ@ݔA~�A�k�A��FA�|�A��A�~�A̍{@�JA�}�A��B�*A�}
@�CmA�e�A��=Ay �A@�A���A�2B��A�}�AOo�A�:0A̒�@C��B	��A5�@���B��B�cA���A��C���>k��A�s�A�k�>��@A���?���A��pA ��C�&�Au��B�kBɘC��yC� AS�@��@���@�\�A%3�A^�@�j�A��A��@�#�A��A�zz      +            '               %   8   	   &         	                                                 I      $         m            !            	      	      
      '      B            3      '   	         .         ,            "   
                                                #   7      -      /            !                                       3      #         =            )                              #      1      !   #   %                           /                                                               !   %      )                                                         -      #         3            )                              #      /      !   #   #                           /                              N-ԾO��N\��N]��N���N���NZ�gN7�,N'C'N��O�%�O��MNߏ�PJ�O	M8O�UN[�OTaM�%�O%�$Ok�UN���Owm�NQ��O�+N(��NCS�NR�N>y:NI%}O!�eNՅ�P}SO;��O�!N��]N'nPc\,N��eNY0N�R�O��(N1Q�Od�uN��N�GmN��N���N��N�=~NO,�O��OLb�PA�N�<DO��O��DO��QN��N׶[N�n�N@�!N1��N닺OT �O<��P�5NM'�N�:"OY�ON�	N�nOz��O<�O-��N�9N��;  �  �   �  Q  h  �    �  q  h  5  �  �  �  �  G  P    �    M  �  �  �  �  �  [  �  U  a  N  2  �  �  �    >  
�  �  �  �  �  �  �  S  b  �  �  Z  ~  2  �  �  	�  �  ,  �  �  �    /  �      T  *    %  �  �    �  �  �  �  �  |<�`B<o<�1<��
<���;�o<t�;��
;��
��`B�o�o�#�
�T���49X�u�D����9X��C������㼓t���C����㼬1��j��j�ě���j��������`B���������o�y�#����w��w��w�',1�0 Ž49X�49X�8Q�ixսD���]/�]/�P�`�]/�P�`�P�`�T���aG��Y��e`B�]/�aG��aG��u�e`B�aG��e`B�m�h�q���}󶽋C���+��+���P��C������� )*)#��������������������()+5>BGB5-)"((((((xz��������zxxxxxxxxx����������������������������������#)/<A@=<:/)#����������������������

�������������FNR[gmst{tslg[XNMHFF���)69:6)������FJTmz|���|smaTNHGFF#/4<@GA</#Taz����������zaXQPRTHKKIHB</-##%/<HHH	*6CIQ_de^OC*��

������������������������������������������������������������������������#/<HQafaUHD</#Y[hqt�������}thh[[YY�������������������������������������������������������������������������������������������������������������������������������������������������


������������#'�������=BCN[grrrg[NKB======}���������������~zy}!#$*/1<HTUSSQOH</#!!��������������������������������)+3)))ht�����������tgaehfh��������������������#--/0/&##+/1<DB</-#�������������������������������������{�����������������~{kty������������tkkkk`amtzz|�zymea^\````��������������������NOX[hsonh[ZOKGNNNNNN��������������������./<CGC><3/'',-......KOZ[]hjhg[UOMHKKKKKK���).(#�������V[^gt|��������thg[XVGUanz�������zp]UJCCG�������������������� %).8IUbnspjaNI<0# t�~}��������������ttfjt�������������tmgfst���������}tmjlssss������	 ����������BFOU[]``_^][POLB:<?B������������������������������������������
!
���������!#/HU_ahfaUOH1/&%#!!\anz����������zna\Z\��������������������������������������������������������
 #0;EME<70#
���)5>CB>5)����w{�����������{yuttwwAEGN[gtvz}�tg[NHB@A��������������������;<IU^hmjiebUKI><656;���������������������������������������޺ֺպԺֺ��������ֺֺֺֺֺֺֺ��
������������
���#�.�/�4�/�-�#��
�
ƧƦƧƬƳƿ��������������ƳƧƧƧƧƧƧùòñìèìù����������ùùùùùùùù¦¦²¿��������¿¾²¦�ּʼʼ����ʼ˼ּ��������������ּ��)�'�)�.�0�6�B�O�R�O�I�B�:�6�)�)�)�)�)�)�U�S�L�K�L�U�a�d�a�]�^�W�U�U�U�U�U�U�U�U���������������������������������������������������������������ĿȿƿĿ�����������l�j�n�|������������̼Լ̼�����������5�(������5�A�Z�g�����������s�g�N�5�/�%�"���"�+�/�;�H�K�N�N�H�B�;�/�/�/�/čăĒęĖĐĚĦĿ��������������ĳč��$�0�=�>�I�O�J�I�D�=�0�$��������;�.�"�	��������	�"�.�;�G�H�D�F�D�F�C�;�M�J�B�M�Y�f�p�r�t�r�f�Y�M�M�M�M�M�M�M�M�ݿѿοǿ¿Ŀпѿݿ���������������ù÷ø÷ù������������üùùùùùùùù�������������������������������������������s�g�Z�V�R�Q�V�Z�g�s�x�����������������/�.�(�#�!�!�#�/�<�H�U�U�U�R�H�H�<�;�/�/�m�a�T�H�@�>�@�H�T�m�z�����������������mìéàÞÙÚàãìóù��ùñìììììì������r�f�c�b�f�r���������������������ÓÓÏÓàììîìàÓÓÓÓÓÓÓÓÓÓ���������������������������������������������������$�$�$��������������������� ���������������������������������������������~���������������������������������������������������������������������������׿Ŀ����~�t�q�u���������Ŀ���������ľs�k�f�Z�P�M�E�C�M�Z�f�s�������������s�n�b�U�J�=�;�B�I�b�n�{ŔŠūůŬŠń�{�n��ŹŹŴŹ�������������������������������I�C�B�H�I�V�b�b�b�Z�V�L�I�I�I�I�I�I�I�I�[�F�A�F�N�[�t¦²��������������²�[���������������ʾ׾����׾ʾ����������n�c�a�Y�a�n�u�z�}ÇÌÇ�z�u�n�n�n�n�n�nàà×ÓÑÓÜàìïöùýù÷ìàààà���������������ɺֺ���!�5�@�-�!���ֺ�������"�$�+�0�0�0�(�$����������������������(�4�<�A�@�9�4�+�(�������������������������������������������ƾƷ�����������������������������������������������������������������������������������������������������������������������������"� �������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��ù����������ù˹Ϲ۹ܹ޹ܹϹùùùùù�ìâàçóù�������������������ùì�������������������������������빝���������ù��3�@�L�J�>�'����ܹù����e�_�`�e�l�r�~�������������������~�r�e�e���������s�c�]�X�[�g�s�������������������o�Y�@�3�*�*�.�;�@�J�L�e�m�w������|�t�o��������������������	��"�2�7�8�5�/�"����������������������������������������D�D�D�D�D�D�D�D�D�D�ED�D�D�D�D�D�D�D�Dӿ����|�y�w�y�������������ĿſĿ����������b�^�a�b�o�{ǀ�}�{�o�b�b�b�b�b�b�b�b�b�bǈ�|�{�w�y�{�}ǈǔǘǕǔǈǈǈǈǈǈǈǈEEEE E*E.E7ECEPE\EYEPEOECE7E*EEEEEiEbEgEuE~E�E�E�E�E�E�E�E�E�E�E�E�E�EuEi�ʾ¾��������ʾӾ׾ھ������������׾ʼf�`�f�����ּ�����������ʼ����r�f�:�1�:�>�F�S�_�l�q�l�_�_�S�F�:�:�:�:�:�:���~�������������������������������������Ľ������������������н۽������ݽнĿ"��������	���"�-�4�8�>�;�:�.�"���������'�4�@�G�K�D�@�4�'���Ěčā�u�}ćĕĚĦĳĿ����������ĿĳĦĚ����������������������������	��������ػ����������ûлܻ�������������ܻлû������������������������������������������(�5�;�;�5�/�(����������� C ( F N F  o � F  7 V 1 -  F : 7 � 0 ? : b Z M L _ g Y N N C E 2 7 . h M S E G L w  b T j B ) _ 5 5 < | S b R R $ * R 4 A I J $ u u O  Q < 8 = ! j      L    �  w  
    �  �  2    �  )    j  +  �  _  [  z  i  �  �  ?  �  i  4  o  X  I  W  |    �  �  �  �  m     �  e  �  g  �  �  �  �  2  �    �  _  �  �  r    �  �  �  �  �  4  \  b    �  �  w  �  �  �  �    �  �  o  �    >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  �  �  �  �  �  �  �  q  G    �  �  P    �  h    �  o    ~  �  )  X  {  �  �  �  �  �  �  �  g    �  �  U  �    R   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   x   p   h   a  Q  G  <  2  '        �  �  �  �  �  �  �  t  Q  .    �  h  f  c  R  @  +    �  �  �  �  r  M  )  �  �  �  �  �  S  5  b  �  �  �  �  �  �  �  �  �  R    �  �  c    �  G          �  �  �  �  �  �  �  �  x  d  T  C  3      �  �  �  �  o  D    �  �  w  >    �  �  U    �  �  e  �  �  3  q  i  `  X  P  K  E  @  @  J  T  ^  _  Y  T  N  G  ?  8  0  :  A  @  E  f  g  b  [  T  J  =  .      �  �  l  &  �  �  '  5  2  *         �  �  �  �  �  }  P  
  �  T  �  ~  �  1  �  �  %  V  }  �  �  �  �  �  T  #  �  �  @  �  �  �  j  z  }  �  �  }  x  q  j  a  W  J  <  -        �  �  �  �  �  �  �  �  r  _  \  [  P  8    �  �  �  q  :  �  �  G  �  �  �  �  �  �  z  d  J  /    �  �  �  �  �  ^  ;    �  �  �  �  D  E  <  -           �  �  �  �  e  2  �  n  �  �  P  F  =  2  &        �  �  �  �  �  �  �  	    !  +  6  �  �  �  �  �  �      �  �  �  �  ~  S  +  �  �    B  �  Z  f  r  ~  �  �  �  �  �  �    �    ?  �  �  G    �  �  �  �  �  �  �  �  �  �        �  �  �  k  /     �  �  M  1  C  L  G  @  A  >  5  ,  "      �  �  �  {  ,  �     �  �  �  �  �  �    j  S  :       �  �  �    U  '  �  �  �  �  �  �  ~  t  k  a  U  G  7  &    
  �  �  �  �  �  �  D  v  �  �  �  �  �  �  �  �  �  {  g  H  (    �  �  �  q  K  �  w  _  J  7  %    	  �  �  �  �  �  �  y  z  �  �  �  A  �  �  �  �  �  y  ^  @    �  �  �  x  I    �  �  �  �  [  [  W  T  P  L  I  E  A  =  9  6  2  .  (      
     �  �  �  �  �  �  �    .  ?  P  ]  j  v  �  �  �  �  �  �  E  �  U  J  ?  3  &      �  �  �  �  �  x  R  !  �  �  x  5  �  a  R  D  5  '      �  �  �  �  �  �  �  �  �  �  q  U  9        N  K  F  A  4       �  �  �  �  ^  &  �  \  �  T  2  $    
  �  �  �  �  �  �  �  �  �    t  n  h  �  �     �  �  �  �  �  �  �  �  �  T    �  {  
  �  �  <  �  �  H  �  �  �  �  �  �  �  �  �  �  �  i  D    �  �  7  �  -  r  �  �  �  �  �  �  �  f  I  +    �  �  �  V  �  �    �          �  �  �  �  �  �  �  �  �  �  �  x  i  K  (    �  >  $  
  �  �  �  �  �  h  K  -    �  �  �  W    �  �  r  	�  
a  
�  
�  
�  
�  
�  
x  
;  	�  	v  �  F  �  ~  �  X  g  �   �  �  �  {  u  o  j  e  a  ]  X  T  O  J  E  @  -     �   �   �  �  �  �  �  �  �  �  �  �  �  z  u  p  n  q  t  w  x  y  y  �  �  �  �  �  �  �  �  �  ~  l  Y  M  F  >  7  +        �  �  �  �  �  r  R  $  �  �  X  �  �  Q  �  �  w  V  ;  �  �  �  �  �  }  q  e  U  C  1       �  �  �  �  �  �  �  �  �  �  �  �  �  t  g  Y  J  8  #  
  �  �  }  3  �  u    �  S  A  /      �  �  �  �  �  x  ]  @  $    �  �  �  �  �  b  W  L  =  +      �  �  �  �  �  [  1    �  �    Q  "  �  r  c  U  F  7  (      �  �  �  �  �  �  ^  =    �  �  �  �  z  h  T  ?  +      �  �  �  �  �  �  �  �  �    B    6  F  M  L  N  Q  W  Z  W  M  7    �  �  V  �  �  ,  �  ~  o  `  J  3    �  �  �  �  ~  X  1    �  �  e  (  �  �  �  �  �  �      .  1  ,  %        �  �  �  �  n  >  �  �  �  �  �  �  �  �  �  g  :    �  �  w  D  �  �  �  �    �  �  �  �  �  �  �  �  o  T  9      �  �  �  [  >  4  2  	j  	�  	�  	v  	^  	2  	  	  �  �  /  �  9  �  �  �  �    �  �  �  �  �  �  �  �  �  v  f  Q  :    �  �  �  ~  Q  "   �   �  ,      �  �  �  �  �    j  \  A  #    �  �  �  \  0    �  �  v  Z  8    �  �  �  �  �  o  L  "  �  �  n  (  �  �  �  �  �  �  �  v  >  �  �  �  �  P    �  n    �  �  ^    �  �  �  �  {  l  [  J  9  &    �  �  �  y  ;  �  �  i    
h    
�  
�  
�  
�  
�  
b  
.  	�  	�  	:  �  H  �    S  �  �  D  /    
  �  �  �  �  �  �  �  l  V  @  0       �  �  �  O  �  �  �  �  �  {  s  j  b  Y  P  E  ;  1  &    �  �  �  �            �  �  �  �  �  �  �  �  r  Z  A    �  �  �  �  �    �  �  �  �  [    
�  
�  
1  	�  	"  W  p  �    n  )  M  D  	  �  �  B  �  r  *  9  �  �  m    �  X  �  �  `  G  *    �  �  �  �  �  f  G  %  �  �  �  l  ,  �  �  y  9  �    �  �  �  �  �  i  H    �  �  E  �  �  %  �  \  �  $   F  %      �  �  �  �  �  �  �  �  �  �  �     D  Q  X  _  f  �  �  �  �  �  �  �  �  {  b  H  -    �  �  �  �  �  �  M  �  �  �  �    v  d  K  0    �  �  �  �  O    �  o  �  a  �      �  �  �  t  0  �  �  L    �  y    �  ?  �  @  �  a  ~  �  �  �  �  u  f  V  D  2      �  �  �  B   �   �   l  �  �  �  o  L  &  �  �  �  u  4  �  �  �  O    �  ~    |  �  �  �  �  �  �  �  �  �  �  �  �  l  ?    �  g    �  !  �  �  �  ~  v  h  [  L  >  3  (        �  �  �  �  �  �  �  �  �  t  T  -    �  �  o  <    �  �         �  �  T  |  `  =    �  �  �  p  8  �  �  {  .  �  �  (  �  -  �  