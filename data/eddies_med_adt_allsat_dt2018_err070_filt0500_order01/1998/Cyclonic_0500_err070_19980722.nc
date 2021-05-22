CDF       
      obs    >   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��n��P      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��t�   max       <�o      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?^�Q�   max       @F���
=q     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�=p��
    max       @v�\(�     	�  *D   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @P            |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Α        max       @�l�          �  4p   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �.{   max       ;��
      �  5h   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�k�   max       B3�a      �  6`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�L�   max       B4T�      �  7X   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?���   max       C��#      �  8P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�~�   max       C���      �  9H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  :@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A      �  ;8   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?      �  <0   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P�{�      �  =(   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�qu�!�S   max       ?۟U�=�      �  >    speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���
   max       <D��      �  ?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?c�
=p�   max       @F������     	�  @   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v��\)     	�  I�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @M�           |  Sp   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Α        max       @��          �  S�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         EF   max         EF      �  T�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�PH   max       ?ے:)�y�     `  U�                     ,                  "         E   
               +      I         	      *      
               1               "   T   �   	   3            ,   A               =   	      1            BN��O�s�N�dFN� Nf��N�IP��OއdNV�M���O�٭Oy�P?!Nk%0OkSP��NM��O�Z
NE�NuP�NM�KP;�OHP0�N�ڎN��ZN��OnT�P3�O��8N�M�O�N�N�N �N��PYN�b�O٠Ofr�O��5O�rP^��O��O��PO�2OxO�lM��O���P�{�O �}O�6�Of�OI"�P�N�.-N!WP",N���N.�*ORU�O�k�<�o;�`B;ě�;��
;�o;�o;D��:�o%   ���
���
�ě��o�#�
�T���T���T���u��t���t���t����
��1��1��9X�ě��ě��ě�����������h�o�\)�t���P��P��P������w�#�
�#�
�',1�0 Ž0 Ž49X�49X�<j�<j�@��D���D���L�ͽL�ͽP�`�]/�ixսixսm�h��o��t�:<=HUakja]UH<<::::::����&'#
�����������������������������������������������������������������������������������pttpt�����������tcipINXZmy{����xvmaTHCAI56BO[_[[OB;655555555��

������������Y\chu������������unY"%+/<FHLJLPOH?<9/'#"������������������������������������NTamvz�����~zmaTQKLN�������������Y[`hjtvthc[XXYYYYYYHUan{}�������}nUQBBH���� ����������������������������~���������������������������������
������������������������������#0IUbggbUG<0
����)66BGB=6)%
#/06/%#
6<@HTUVaea^^XUH<:666;BINUahmqnibXUI<978;�����5=:)��������)67BHTZ\XOB)�����������������������������������������������������;<>GIU\[UKI<;;;;;;;;aacfjnz~~{znhaaaaaz����������������}sz��������������������ms�������������zmhgm��������������������BNq{��}|yxtgf[WB869BAGNSgt����}gNC<9:9;A-/>HLkrv{{naU:#���
 /=GHE</#
�����t~��������������wqmtCRgt���������gTNL>;Cstv�������������yts����������������������������������������������� �������������&315AGE)�����fhpt�����������tohef������������������������!������nv{������������{nkin������#���������*56BNQ[\][SNB85)****GHPTTTSMHDCCGGGGGGGG��(8BNggZB5)�����gtt�������ytgggggggg��������������������������
 ���������hkpx�������������thh�A�;�4�,�*�.�4�A�H�M�W�W�M�K�A�A�A�A�A�AàÇ�z�w�zÀÇÓìùÿ������������ùìà¦±²³³·´²¦¦�T�L�M�G�G�G�P�T�`�m�t�y���|�y�m�e�`�T�T������{�s�r�s��������������������������(�%� �%�(�4�A�D�M�Y�U�M�A�4�(�(�(�(�(�(�ʾ��f�Z�M�C�G�R�e����׾���	����	���[�O�B�6����6�B�[�dāčġĥĦĘčā�[�;�8�2�6�7�;�G�H�S�Q�G�>�;�;�;�;�;�;�;�;���������������ȼż����������������������ʾƾ����������������������˾־ݾ�߾׾��B�5�)�#������)�5�B�D�N�V�[�g�[�N�B�`�K�8�:�,�(�.�;�R�_�e�r�������������y�`��������������������������
����"�;�A�H�L�T�\�[�X�M�H�/�"�����������������	�;�m�z�������z�a�;�"������������»ûлһӻлɻû��������������-����	�"�H�a�z�������������m�a�H�;�-���������������������������������������������������������������������������������N�H�A�?�A�L�N�Z�c�g�k�g�^�Z�N�N�N�N�N�N�*���������Żž������*�F�h�s�r�h�S�C�*�������������ɺֺ���������ֺܺɺ��������y�q�r�y����������������ݽн��a�W�T�H�G�E�H�H�N�T�Y�a�f�k�m�p�m�f�a�a�B�@�6�4�1�5�6�B�J�O�O�U�U�O�B�B�B�B�B�B���
��
��
��
���#�/�3�6�7�/�#���(���� ���4�A�M�Z�`�f�j�q�e�Z�M�4�(�T�.���־;ھ���	��3�B�T�m�����|�m�T���t�f�Z�a�g�n�s�������������������������Z�U�P�O�Z�f�s�u�����~�s�f�Z�Z�Z�Z�Z�ZùñìàßÚÚÚàìùÿ����������ûùùàß×Ôàìùúÿ������ùìàààààà�����������������������������������������A�<�5�(�����(�5�A�N�O�S�N�G�A�A�A�A����ɺ����ֺ��!�S�l�x���������l�_�F�-��лʻû������������ûлܻ�����ܻл�ƳƧƏƃ�xƎƳ�������	����������������Ƴ����������������*�6�C�M�N�C�6�*������	��*�6�C�O�\�h�uƁƋƆƁ�p�h�N�6�*��	����������������������$������������E�E�E�E�E�E�E�E�FF1FJF|F�F�F�FVF3FE�E�ED�D�D�D�D�E
EEE*ECEZEaEbE^EVEPE*EEł�{�u�{ŇŎŔŠťŹ��������ŹŸŭŠŔł¿¦�y�}������������
��
����¿�<�4�0�#��"�#�.�0�<�I�T�U�b�b�b�_�U�J�<�L�D�L�O�Y�g�~�����������������~�z�e�Y�L�лͻͻ˻лٻܻ�߻ܻллллллллл�����ĽĺĿ�����������
��$�'�&�#������Ƚ!���
��!�.�y�����Ľн��	���̽��y�!��������'�*�3�@�G�B�@�<�7�3�-�'��d�Z�Y�X�_�h�s�����������������������s�d�4�2�.�-�2�6�@�M�R�Y�f�t����y�l�Y�M�@�4�4�'�"�����/�@�M�Y�\�^�_�b�c�_�Y�@�4�f�c�r�������ؼ��������ּ����r�f�Ľ����������½ĽнԽݽ��ݽڽнĽĽĽ��g�[�g�r�s�t�����������s�g�g�g�g�g�g�g�g�u�y�w�a�H�3�/�<�H�níùþÿøèæÓÇ�u�<�/�/�$�&�/�<�H�Q�L�H�B�<�<�<�<�<�<�<�<���y�x�s�v�x�����������������������������������$�0�=�I�V�X�Y�V�N�I�=�0�$�ā�t�h�[�O�H�C�B�C�O�hāčĞĥĩĪĭĦā 7  J 0 b 2 B b U W [ Y u R . W e q E 8  U K D O ^ 7 1 b D ( = G d W � M T Y ^ E ] * h = ( t p ! M ] G * 1 s 9 w h T N 5 9  �  5    �  �  �  �  L  �  9  ?  m  �  {  �    �  �  <  {  Y    E    �  �  �  �  �  a  �  D  �  V  �  �  +  R  #  �  ?  F  ,  c  �  /  �  I  �  ]  =  D  �  �  [  �  i  �  �  Q  �  ;��
�u�T����o;o�D���#�
��h��o�o���
�e`B�'D����`B���T��9X��h��1�ě���9X�}��w�����`B��h�o�T����7L�T����w�Y��<j���'� Ž]/�ixսixս�O߽��P���m�.{�P�`��vɽ@��T���<j��Q��G��y�#��\)����}��S��q���ixս�
=��\)�}󶽲-�I�B�CBhmB�B!(B�kB��B�A��[B�B$,SB3�aB�"B*�]B*<5A��%B�<B��B?�B�B4�BƨBQ�B!qZB%�B6wBuCB@�B''B��B�B��B�gBQbB&ʺB�BQ�B4B'zB��BĐB�tB�B��B
�kB
�B
�2B�sB#=�B��B�B�yB�BqyB)f+B-��B�rA�k�BP^B
g�B!,�B³B/�B��BB$B:gB!A�B��B�LB��A�MBQB$+B4T�B�B*�SB*�A��SB��B��BrLB�UB�sB��B@QB!@VB%�DB�B��B)�B'UB:MB��B��B��BH`B&��B�FB?�B=�B @�BM�B��B�6B�B�B
@_B
3QB
�vB��B"��B?�B<BA%B9�BB�B)w�B-��B�A�L�BayB
��B!�B�>B@�A:��A�*�A��Ai@AFcA9ʐAL�#A���Ac��@�AN��A�YAk�A�C�A��OA��9@��,A��A�1A�k7A�$A���@7��A$��A��A؟�A�)�A:$�A`U�A��!AA.�A��A̲�A���A�\p@w��@�ݦB�~A��B ��A�ͳC��#C���A��A�ǵA��b?�kF@���A�ɟA��?���A�0�@״}@��TA *�A( BA���A��DA�e�@��FB
O�A�}�A:�*A�A�O�Aj��AE�1A9�
AP&A�l�Ad�6@�#0AM��A���Ak�A�:A���A�h-@��A��1A�@�A�lA�dB 0�@3�A"�A�quAؘFA�A:��A]�A��nAB~A�AÃ�A�<A�|�@t!�@�)B@A���B �XA��`C���C���A�8A�||A�v�?��*@�FA�{�A1X?�~�A���@��2@��>A��A(�vA�>�Aʈ!AÊe@���B
�3A�{�                     ,               	   "         E   
               +      J         	      *                     2               #   T   �   
   3      	      -   A               >   	      1            B                     ;   %               1         A      +            1      +               1   !                  1      '      !   %   9   #      -               ?               1         /            #                     /   !                        5      +            !      +               +                           '            /         +               ?               '         -            N@UO�\	N�z�N��Nf��N�V�P6XO��FNV�M���O?�-N菗O�CNk%0O �9P��aN#��O�Z
NE�NuP�NM�KO�ȍN��P��N�ڎN��ZN��Oc-"Py�O_�$N�M�O�N�N�N �N��N��N�mrO٠Ofr�O@��O���P@6O�N\O��P5OxO�lM��O]P�{�N�׆Ox?OH�OI"�O�4�N�.-N!WP�cNa��N.�*OELO�\!  T  �    �  �  �  �  �  �  
  �  �  �  J  4  �  i  �  3  ^  �  �  �  �  Y  �  6      �  *  s  �    7  }    �    �  �  �  �    �  v  �  �  �  �  �    �  �  "    �  ]  �  7    
!<D��;ě�;��
;�o;�o;D���ě��ě�%   ���
�o��`B��/�#�
��t����ͼe`B�u��t���t���t��+�ě��\)��9X�ě��������ͼ�����h�o�\)�t���P�����w�����H�9�H�9�@������,1�@��0 Ž49X�49X�m�h�<j�D���H�9�L�ͽL�ͽe`B�P�`�]/�m�h�q���m�h������
CHU``VUHF@CCCCCCCCCC����!%&#
�������������� �����������������������������������������������������������������������}���������������trr}EKNV\akmz}���~zmaTHE56BO[_[[OB;655555555��

������������hiku��������������zh,/8<HHKNJH><7/)%#',,����������������������������������������QTWahmwz{{|zmaYTSQQQ���������������[[]hhttzthe[YZ[[[[[[HUan{}�������}nUQBBH���� ����������������������������~���������������������������������

�����������������������������#0<IYacb\UI<0
����)66BGB=6)%
#/06/%#
8<DHNUX[UUH=<7888888<CIOU_bhlpmhUI<9889<����)55)��������)*6BEPWXUOB6)"����������������������������������������������������;<>GIU\[UKI<;;;;;;;;aacfjnz~~{znhaaaaa���������������������������������������ms�������������zmhgm��������������������>BGN[htyyttogc[UNA>>BNS[gt���~tg[NEA>>?B/<UbgnsxxuaU4#���
#)./0-#
�����t~��������������wqmtFUgt���������gZNB>=Fstv�������������yts��������������������������������������������������������������&315AGE)�����fhst~����������tqhff��������������������������������nv{������������{nkin����� ����������*56BNQ[\][SNB85)****GHPTTTSMHDCCGGGGGGGG��)9BNdfZA5)�����stw�������~tssssssss��������������������������

����������s{�������������tmlns�4�0�/�4�A�M�O�O�M�A�4�4�4�4�4�4�4�4�4�4àÇ�z�x�zÃÇÓàìùþ������������ìà¦²²¶´²¦¤�T�R�Q�J�T�[�`�m�r�y��y�w�m�b�`�T�T�T�T������{�s�r�s��������������������������(�'�"�&�(�4�?�A�M�Q�Q�M�A�4�(�(�(�(�(�(���a�S�R�Z�f�z���������ʾ׾�������׾�ā�h�[�O�B�4�)�"�'�6�B�L�[�h�qĒĜĝďā�;�8�2�6�7�;�G�H�S�Q�G�>�;�;�;�;�;�;�;�;���������������ȼż����������������������ʾ����������������������žʾҾ׾۾ݾ׾��)�%�����)�5�B�B�N�S�[�d�[�N�B�5�)�)�y�m�`�_�T�R�I�Q�T�`�m�q�y�������������y������������������������"������"�+�/�;�H�J�N�O�H�>�;�/�"�"�������������������	�"�H�m�y���z�a�K�"�������������ûûлѻһлȻû��������������-����	�"�H�a�z�������������m�a�H�;�-���������������������������������������������������������������������������������N�H�A�?�A�L�N�Z�c�g�k�g�^�Z�N�N�N�N�N�N�*���������������*�=�^�f�e�\�O�C�*���������������ɺֺ�������ֺպɺ��������x�w�z���������нܽ޽۽ܽ��ݽн��a�W�T�H�G�E�H�H�N�T�Y�a�f�k�m�p�m�f�a�a�B�@�6�4�1�5�6�B�J�O�O�U�U�O�B�B�B�B�B�B���
��
���#�/�0�3�/�.�#�������(��������4�A�M�Z�h�o�c�Z�M�A�4�(�.����Ӿ�����	��-�=�T�`�t�y�m�`�G�.�s�g�g�j�q�s���������������������������s�Z�U�P�O�Z�f�s�u�����~�s�f�Z�Z�Z�Z�Z�ZùñìàßÚÚÚàìùÿ����������ûùùàß×Ôàìùúÿ������ùìàààààà�����������������������������������������A�<�5�(�����(�5�A�N�O�S�N�G�A�A�A�A�!�����!�$�-�:�F�S�T�S�K�F�@�:�-�!�!�лϻû������������ûлܻ�����ܻл�ƳƧƏƃ�xƎƳ�������	����������������Ƴ����������������*�6�C�M�N�C�6�*������*�!����*�9�C�O�\�`�h�p�h�b�\�U�C�6�*���������������������������������E�E�E�E�E�E�FF1FJFoF|F�F}FJF0FE�E�E�E�EED�D�EEEE*E7ECEPEWEYEVEPEKECE*EEł�{�u�{ŇŎŔŠťŹ��������ŹŸŭŠŔł¿¦�~¦¿�����������������¿�<�4�0�#��"�#�.�0�<�I�T�U�b�b�b�_�U�J�<�L�D�L�O�Y�g�~�����������������~�z�e�Y�L�лͻͻ˻лٻܻ�߻ܻллллллллл��������������������
�������
�����ؽ!���
��!�.�y�����Ľн��	���̽��y�!����	�����'�+�3�@�E�@�:�6�3�+�'��s�g�[�Y�`�g�i�s�����������������������s�@�=�4�2�0�4�9�@�M�Y�f�r�~�w�r�j�f�Y�M�@�4�'�"�����/�@�M�Y�\�^�_�b�c�_�Y�@�4�����������ܼ������������ּ������Ľ����������½ĽнԽݽ��ݽڽнĽĽĽ��g�[�g�r�s�t�����������s�g�g�g�g�g�g�g�g�v�y�x�a�H�4�0�<�H�nìøþÿ÷çåÓÇ�v�<�;�/�'�)�/�<�H�O�I�H�<�<�<�<�<�<�<�<�<���y�x�s�v�x�����������������������������������$�0�=�I�V�W�Y�V�M�I�=�5�$��h�[�O�F�E�F�O�h�tāčĘğĢĢğčā�t�h 3   = 8 b . > c U W Z K F R - \ j q E 8  > D F O ^  0 ` 7 ( = G d W N K T Y D & ^ 1 h : ( t p   M ` B & 1 W 9 w h - N 4 0  U  !  �  �  �  �  b  }  �  9  �    K  {    �  z  �  <  {  Y  �    {  �  �  �  �    �  �  D  �  V  �  �    R  #  �  .  �  
  c  C  /  �  I  �  ]  $  �  �  �  o  �  i  �  w  Q  �  �  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  EF  B  F  J  M  P  R  S  T  S  O  D  :  0       �  �  �  J  �  �  �  �  �  o  L  $  �  �  �  v  U  .  $    �  �  �  T  �  �       �  �  �  �  b  3    �  �  a  '  �  �  K  �  :  Y  �  �  �  �  �  �  �  �  �  �  �  |  v  o  j  e  _  [  W  S  �  �  �  �  �  �  �  �  �  �  }  {  x  v  s  q  n  k  i  f  �  �  �  �  �  �  �  �  �  �  �  �  x  i  V  @  *    �  �    `  �  �  �  �  �  x  a  C  %    �  �  �  �  �  `  �  c  �  �  �  �  �  �  �  g  8  �  +  %  �  �  ]  !  �  =  =  �  �  �  �  �  �  �  �  �  �  �  �  |  o  a  T  C  2  !     �  
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  e  W  I  �  �  �  �  �  �  �  �  �  {  j  W  H  8    �  �  �  |  _  �  �  �  �  �  |  f  O  1    �  �  �  �  y  n  h  G   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  X    �  1  �  J  E  @  :  5  0  +  %              �   �   �   �   �   �   �      *  0  1  2  4  2  ,    
  �  �  �  �  a  ?        q  �  �  �  �  �  �  �  �  y  6  �  �  Q    �  !  �  �  �  U  _  h  ~  �  �  �  �  r  U  7    �  �  �  �  d  =    �  �  �  �  �  r  �  �  l  T  <  $    �  �  �  �  �  �  l  G  3  *  !        �  �  �  �  �  �  �  �  �  x  e  Q  >  +  ^  T  I  ?  7  .  &        �  �  �  �  �  �  q     �   \  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  b  Q  @  )    F  �  �  �  �  �  �  �  �  f    �  X  �  ~  �  �  �  �  �  �  �  �  �  �  k  J  !  �  �  g  1       �  i    �  .  h  �  �  �  �  z  V  "  �  �    �  �  /  �  
  '     �  Y  G  5  #    �  �  �  �  �  �  �  �  z  f  Q  =  )      �  �  �  �  �  �  �    r  o  l  i  [  E  /    �  �  �  L          +  4  ,  $        �  �  �  �  �  �  �  �  �      	      �  �  �  �  �  w  G    �  �  J  �  �  E            
  �  �  �  �  �  �  �  �  n  8  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  O    �  �  %  �  �  K  0  *  '  $          �  �  �  �  �  �  \  '  �  �  �  �  �  s  g  \  N  A  5  3  1  '    �  �  �  Z  !  �  �  Q    �  �  �  �  �  �  �  z  h  V  B  ,    �  �  �  �  S    �  _         �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �  7  +        �  �  �  �  �  �  �  �  �  �  �  �  �  {  t  �  �  �  �  i  u  v  n  c  n  u  z  D  �  �  J  �    .  S        	  �  �  �  �  �  w  \  @  !  �  �  �  f  3  �  �  �  �  �  �  �  �  �  m  F  &  	  �  �  �  j  8    �  �  x    �  �  �  �  �  s  Q  5  A  2    �  �  �  S    �  �  �  �  �  �  �  �  �  �  �  �  �  �  X    �  i    �  ,  i  �  ?  @  E  f  ~  �  {  j  O  &  �  �  s  )  �  �  3  �  W  j  O  �  �  �  p  ,  
�  
x  
  	�  �  1  {  �  �  �  �      �  �  �  	  h  �  �  �  �  �  `  (  �  h  �  �  �  q  
   n  l      �  �  �  �  �  �  �  �  �  �  �  d  :        *  <  �  �  �  �  �  �  {  K    �  �  �  �  I  
  �  |  9  �  �  v  n  f  ^  W  M  >  .         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    m  \  L  :  '    �  �  �  y  O  #   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  �  �  �  �  �  �  �  �  �  �  v  3  �    �  g  �    S  �  q  H    �  �  h    �  k  �  �  �  k    �  !  �  d  2  �  �  �  �  �  �  �  �  e  A  ?  [  R  J  C  @  \  }  �  �  �    �  �  �  �  �  �  �  �  \  1    �  �  J    �  �  6  �  �  �  �  �  s  a  K  -  	  �  �  �  X  %  �  �  �  {  7  �  �  �  �  �  �  �  �  �  �  n  W  :    �  �  �  s  /   �  �      �  �  �  �  �  �  z  G     �  E  �    k  �  �  z    z  v  q  j  b  V  J  <  /      �  �  �  �  �  ^  -   �  �  �  �  �  �  �  �  �  y  c  N  8  #    �  �  �  �  �  �  ]  \  O  6    �  �  y  �  {  S  &    �    V    m  �  �  �  �  �  �  �  �  �  �  �  �  s  _  H  .    �  �  �  T  #  7  /  '              �  �  �  �  �  �  �  �  �  �  �      �  �  �  �  e  7  	  �  �  o  3  �  �  Q  �  x  
  �  	�  
  
   
  
  	�  	�  	�  	x  	O  	  �  Y  �  @  �  �  �  Y  Y