CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�z�G�{       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��#   max       P�?\       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <e`B       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�   max       @Ftz�G�     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @v������     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P            �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��            7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <o       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B25�       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B2�       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =�g�   max       C���       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =Ƞ   max       C��s       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          P       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��#   max       P�GY       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�(�\)   max       ?Ӥ?��       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �� �   max       <e`B       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�   max       @Ftz�G�     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�    max       @v�Q��     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @N            �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�ɠ           Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F   max         F       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?b��n/   max       ?����E�     �  ]      5      M   
   	         *      *      "      "   	      #   P               M      7                  "                     )      	   3               ,      	                                    +                     *      NC�PBxNTB�P�?\N�MN2�N�oO:SO�h�O�E_P?�jOx�OJO�Z$O�ߔNf%�N!:O��2P2WlN/wyO�qRM��#Nf�O�yQO5i`O�g�O��Ob.N�!�O<|O^QP%mwO"XN���Od�N�lO��iO���Pn*�O�*FN���OضiOV�O�]rN�bqNE>�O�#<O\�GNz��OYe(N�#	O9e�O��N,��N��$N�NNH1UN��Oz�O[ ,OC�&N��	N��N�O�tN�8�N3�BO�?�N�~�N$ �<e`B:�o:�o%   ���
��`B�o�t��#�
�#�
�#�
�D���D���D���T���T���e`B�e`B��o��o��o��C����㼣�
��9X��9X��j��j���ͼ���������/��/��`B��`B��h���+�+�+�C��C��C��\)�#�
�'8Q�<j�<j�@��L�ͽL�ͽT���Y��Y��m�h�q���q���u�}�}󶽉7L��C���O߽��P���P���P�������
�������������������������
/UZaZH/#
�������-6BOT[_[OB;6--------�#In�������`I< ���
 # ���������������������./:<HSOJH=<:0/-+....��#/<HMU[QH/#
��v{������������{rpqtv&-16OZbkjh[XO6)���������������������
#%+*"
������y����������������zpy������������������mptz�������������zqm����������������������������������������������������������3@N[t���������tNB3-38<HLNLH=<78888888888)21-6BO[b`fjh[B6&"()"")/144//.'"""""""""������������������������
#*.-.-#
����� #/7<EHMSUMH<:/'#  `nz����������|ujaUY`�!	�������������������������JOW[hht{trkph[OIDHJJ�����������������������������������������������������������������������������������������������������������������������������������������������

������������������������������������������������}������
�����������
�������x������������������x��������������������BN[gt��������tgNFB9BLO\hu����uh\OMLLLLLL
#'%#
	at�����������~|hc]]a�#08>?@@><0#
��:<BIKNOKI<:656::::::EILNRSUbnswyytnbULIE����������������������������������������37GN[gt������tg[NB83dgiqt���tgdddddddd,0<IKUVURI<0%(,,,,,,��������������������


	









 #&&# 
	
cght�����������tigcc����������	�������������
�������������������������������������������������<DHMPH<:98<<<<<<<<<<_ainz��������zunaa\_�����������
 









��)BNSNKA,������	��������)558651)!���������������������������������������*� ��$�(�A�Z�y�y���~���f�Z�M�A�4��a�]�V�\�a�i�n�t�z�s�n�f�a�a�a�a�a�a�a�a�������`�C�1��1���������������������������������������������������������������������!�,�-�/�.�-�,�!��������6�5�6�6�>�B�O�[�]�h�j�h�[�X�O�B�6�6�6�6¿·µ·������������������������������¿�����������������	�"�3�>�?�8�/�"��	�����7�"�	�������	�"�H�T�]�h�m�s�y�t�m�a�T�7�`�G�2�'�(�$�0�;�T�m�����������������y�`ìáÍÍÙÝßàìù��������������üùì�H�D�>�?�;�<�E�H�U�a�i�n�p�u�k�c�b�^�U�HàÓÎÇ�z�v�r�uÇÓàì��������üùìà�)�����������)�B�O�T�[�v�|�h�[�H�B�)�U�J�H�B�E�H�U�`�a�m�b�a�U�U�U�U�U�U�U�UFFFFF$F1F6F1F/F$FFFFFFFFFF��������������'�4�M�Y�_�a�Y�M�4�'�����������w�r�t�y�����Ŀ��������ݿĿ����������������������������������������ſ6����������;�G�`�y���������y�`�G�6��������������������������������������������������������������������������������E\ECE*E!EEEE*E7ECEPEfEuE�E�E�E�E�EuE\FFFFFFFF$F=FJFVFZFXFVFLF=F1F$FF���������������ùܹ���
�����ܹ¹�����������������������������*�.�*��������������������ûлܻ�����ܻлû����ù������������ù˹Ϲܹ�����ܹϹù������������������������$�/�,�$�$�����5�.�(�$�3�A�N�Z�g�s���������s�g�Z�N�A�5�u�O�6������*�C�\�hƚƳ��ƸƫƧƎ�u�S�A�:�5�:�F�S�_�l�x�������������x�l�_�S�n�g�b�Z�^�b�i�n�{ŇŔŔŔŇ��{�n�n�n�nŠŔŠŬž��������������������������ŹŠ��������������&�*�-�*�$�������������Ŀѿݿ߿�����������ѿĿ��������y�m�i�e�h�m���������������������������r�L�7�=�C�Y�o���ɺ����� ����ɺ��������������Ѽ����� �)�&����ּ������������׾־׾������	���	��������������ݾ޾���	��"�;�E�J�H�A�������M�=�4�-�2�@�K�M�Y�f�k�{�������p�f�Z�M��������������������)�0�3�.������޾������������������žƾȾľ�������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D߻�������������лܻ�� ��ܻл������������Ľ������������������ĽнݽݽֽܽڽϽɽĽ����������(�+�*�(����������������û����������y�u�y�������ûлۻ�޻лȻ�����ĿĿľĿ����������������������������ŔŇ�{�n�n�{�}ŇŌŔŠŭŹ��������ŹŭŔ������»¦¦²������� ���	�����I�D�<�0�,�0�<�A�I�K�P�K�I�I�I�I�I�I�I�I�!����!�-�-�/�:�D�E�F�:�-�!�!�!�!�!�!�
�������������������� �
������
�
�S�Q�O�S�_�l�x���x�v�l�_�S�S�S�S�S�S�S�S�����������������ûлڻܻ��ܻлû������<�0�/�(�)�(�/�0�<�H�U�W�a�U�P�M�H�F�<�<�$�������������$�0�=�B�I�I�7�%�%�$���������������ʼּ��������Լʼ����0�)�$�$�0�=�I�I�V�[�X�V�I�=�0�0�0�0�0�0�ܹӹԹܹݹ�������������ܹܹܹܹܹ�D{D�D�D�D�D�D�D�D�D�D{D{D{D{D{D{D{D{D{D{�Ŀ����������������Ŀѿҿڿܿݿ޿޿ݿѿĽĽ��������Ľнݽ��ݽнĽĽĽĽĽĽĽĽ�����������������������������������òéßçéìñù�����������������O�N�E�O�U�[�f�h�q�t�y�y�t�q�h�[�O�O�O�O���������������������������������������� [ B t A 0 _ < x $ ; / 0 h E C 7 H 3 ; Q f o 0 V O + t 4 1 ] F W W e � ' : > T j F 1 $ 5 w F t * M P b = E E < @ B e F B ; 0  s G + @ c 2 S    �  �  P  �  �  �    �  9  *  �  �  )    z  G  �  +  M  �  =  (  �  �  3  �  0    �  �  V  �  �  �  �  g  �  P  �  �    �  �    ]  �  �  �  �  �  �  E  W  �    h  8  ?  �  �  �  �  b  J  �  >    �  o<o�T���u���-�u��o��9X�����P�`�\)�P�`�t��8Q�'<j��1���
�D���\��t���㼛�㼬1�ě��P�`���-���t���w�49X�8Q�y�#�0 Ž\)�#�
�t��y�#�ixս��P�T���,1��1�e`B��+�8Q�e`B��E�����]/�u�ixս�%��m�h�u��O߽�%��hs���P���T��
=���-�������P�� Ž�9X�����vɽ�1B�vB��Bd�B';�B��B!x>B�)B�B�B	�B+��B��BM�B+fB � B �RB4B(@B	*�B�BI4A���B!c�B�B{*B�BB��B̌BXWBmYB�B��B�FBO�BlB�<B�B�B�[B,��BP�B��B �B	8"B25�BDsB�/B%<B&lB'�=B�:By/B	U�B	�WB&MB�B$�HB$�B
KB&>B��B0%B! B�BK{B�BvhB{BeB� B��BG�BA[B'=]B�"B!AB��B@BASBͳB+A�BE�B3B��B �^B �B>�B7�B͙B>�BB>A���B!iB�pB>�B��B?�B��BApB��B<uB��B�sBH�B@�B��BFB�(B�SB-ƸB?NB��B�B	ARB2�B?�B��B$�FB&��B(?RB?B��B	��B	�B&zzB��B$HzB$��B
��B?�B@CB!�B ��B�rBE�B9BClB��B��BٙA�͊A<A��A�ŴA�U8@gWEA�>�A���A�IA��Ai��A�1<A��A˵TA�i�A�B}C�Ŝ@�TAxwA�f�Ad�=A��gAM�C�� C���=�g�A�E@�ND>�;BN�A�%B@�t�A�8�A�=LA�j�A|�OA�A�@�A~�AW�nA\�
@���A��AN DC�$�@��A$�A2�6@�"A�F�A��zA��6A��5@v�WA�`@��@���A�՚B	�@�ZKB
҂?�vC��CAyPvA(�mA0�IA�qA�^A�0�A��A<!AƁ�A���A���@dF,A�}�A���A�SzA���Ai��A̫�Aƀ�A�|�A�ZA��C��@���Ax�=A���Ak�A��=AL�sC��C��s=ȠA�b9@�#)>�S^BJxA��B��@��A�\A�A�MAz�A��@��A�AX�4A\��@�q�A�.LAN(kC�'�@�
A#�qA4�@�tA�~AA�GXA�E�A�@t�A�@���@���Aá�B	I�@��B<?��C��Ay�A(��A0��A�f�A�~-A�a�      6      N      
         *      *      "      "   	      #   P               M      8                  #                     *      	   3                -      	                                    ,                  	   *            )      A                  %   +            %            +      )               #                  -                  #   7   +      %      !         !                  %                                             #                  5                                 %            !      )                                 '                     7   +                                       #                                                   NC�N��SN8T�P�GYN�N|N�k�O:SOLy�O��=O~�OE��N��IOLDCO�ߔN0c�N!:Of�VO��N/wyO�qRM��#Nf�O&I�N��Ol�O��N��N]eqO<|N��RP
}O-N���N���N�lO�I�O.p�Pn*�O�*FN���OO9�OHeOO&�ON�bqNE>�N�\�OraNz��O@��N�#	O!:�O��N,��N��$N�-�NH1UN��N�_1O[ ,O<K�N��	N��N�O�tN�8�N3�BO|<mN�~�N$ �  %  5  �  .  �  $  �  4    �    �  w  Y  >  �  �  [  	�  �      �  �  �  V  F    �  �  �  ,  �  0  C    �  T  h  i  �  :  �  �  �  '  J  �  �        H  l  >  �  U  _    /  	�      v  Y  �  D  	8  �  E<e`B�C�%   �u�ě��o�t��t�������o�o��o���ͼ�C��T���e`B�e`B��9X�+��o��o��C�����D����h�#�
��j�ě��������o����`B��`B����h�o�,1�+�+�C��Y��\)�H�9�#�
�'u�P�`�<j�D���L�ͽP�`�aG��Y��Y��y�#�q���q���y�#�}󶽁%��7L��C���O߽��P���P���P�� Ž��
�����������������������!#,/<HINLH=</#!!!!!26BOS[^[OB=622222222#0Un�������U<0
"���������������������//2<HPMHH<<<1/.-////��#/<HMU[QH/#
����������������������'-6BOSX[^b[VO6)����������������������
#('#
����������������������������������������������mptz�������������zqm������������������������������������������������������������4:BN[g�������tgNB;648<HLNLH=<78888888888)21-6BO[b`fjh[B6&"()"")/144//.'"""""""""���������������������
#&%#
������##/0<?EHHIH<3/*&#! #pyz������������ztolp�!	�������������������������NO[hrmh][VOLNNNNNNNN�����������������������������������������������������������������������������������������������������������������������������������������������

������������������������������������������������}������
�����������
�����������������������������������������������KNX[gtw����ytgd[ULJKLO\hu����uh\OMLLLLLL
#'%#
	��������������������#'059<;0%#!:<BIKNOKI<:656::::::IMPRUWbnrtvxyxsnbUMI����������������������������������������5:INW[gt������tg[NB5dgiqt���tgdddddddd,0<IKUVURI<0%(,,,,,,��������������������


	









 #&&# 
	
dgkt�����������tlgdd����������	�������������	�������������������������������������������������<DHMPH<:98<<<<<<<<<<_ainz��������zunaa\_�����������
 









���)5=A:53)�����	��������)558651)!�������������������������������������A�?�4�1�2�4�4�A�M�P�Z�Z�^�[�Z�M�A�A�A�A�a�_�W�]�a�j�n�s�y�r�n�c�a�a�a�a�a�a�a�a���k�`�I�B�F�U�g����������������������������������������������������������������������!�-�-�-�*�!����������6�6�6�7�?�B�O�[�[�h�h�h�[�T�O�B�6�6�6�6¿·µ·������������������������������¿�����������������	��!�"�-�,�&�"��	�����C�/�"�����	��'�/�H�S�a�j�s�m�c�a�T�C�`�T�H�=�9�8�<�G�T�`�m�u�����������y�m�`ÓÓÝàáãìù����������������ùìàÓ�U�I�J�M�U�W�a�f�j�j�a�_�U�U�U�U�U�U�U�UëàÓÒËÇ�}�x�zÁÓàì��������ùïë�)�����������)�B�O�T�[�v�|�h�[�H�B�)�U�L�H�B�F�H�U�]�a�d�a�_�U�U�U�U�U�U�U�UFFFFF$F1F6F1F/F$FFFFFFFFFF���	���#�'�4�@�M�Y�Z�Y�R�M�@�4�'���Ŀ������������������Ŀݿ�������ݿѿ����������������������������������������ſ6����������;�G�`�y���������y�`�G�6��������������������������������������������������������������������������������E6E*E$E&E*E6ECEPESE\EfEiEsEuE|EuEiE\EPE6FFFFFFF$F1F6FJFKFRFJFHF=F1F)F$FF�������������������ùϹܹ߹��ڹϹù���������������������������*�.�*����������������ûлܻ�����ܻлû��������ù¹����ùϹ׹ܹݹܹڹϹùùùùùùù������������������������$�/�,�$�$�����N�B�D�N�T�Z�g�s�z�~�{�s�g�Z�N�N�N�N�N�N�5����*�C�\�hƆƚƪƳƲƧƞƎ�u�\�O�5�_�W�S�C�:�9�:�F�S�_�l�x�����������x�l�_�n�g�b�Z�^�b�i�n�{ŇŔŔŔŇ��{�n�n�n�nŭŧŭůŹ����������������������Źűŭŭ��������������&�*�-�*�$�������������Ŀƿѿݿ�����������ѿĿ����z�w�m�l�m�o�x�z�����������������������z���r�L�7�=�C�Y�o���ɺ����� ����ɺ��������������Ѽ����� �)�&����ּ������������׾־׾������	���	������������������	��"�.�3�:�8�.�"��	���?�4�3�@�E�M�Y�f�j�z�������r�o�f�Y�M�?��������������������� ��������꾱�����������������žƾȾľ�������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D߻������������ûлܻ����ܻڻлû������������������������Ľ̽нѽнͽĽ������������������(�+�*�(����������������û����������z�{�����������ûлٻ�ܻл�����ĿĿľĿ����������������������������ŠŔŇ�{�{ŀŇőŔŠŭŶ��������žŹŭŠ������¾²«¦²¿���������������I�D�<�0�,�0�<�A�I�K�P�K�I�I�I�I�I�I�I�I�!����!�-�-�/�:�D�E�F�:�-�!�!�!�!�!�!���������������
�����
��������������S�Q�O�S�_�l�x���x�v�l�_�S�S�S�S�S�S�S�S�����������������ûлڻܻ��ܻлû������<�3�/�*�,�)�/�3�<�H�R�U�_�U�O�L�H�D�<�<�$�������������$�0�=�B�I�I�7�%�%�$���������������ʼּ޼������Ӽʼ����0�)�$�$�0�=�I�I�V�[�X�V�I�=�0�0�0�0�0�0�ܹӹԹܹݹ�������������ܹܹܹܹܹ�D{D�D�D�D�D�D�D�D�D�D{D{D{D{D{D{D{D{D{D{�Ŀ����������������Ŀѿҿڿܿݿ޿޿ݿѿĽĽ��������Ľнݽ��ݽнĽĽĽĽĽĽĽĽ�������������������������������������øíèäèìù��������������������O�N�E�O�U�[�f�h�q�t�y�y�t�q�h�[�O�O�O�O���������������������������������������� [ = m 9 ( V @ x & : F / g J C B H %  Q f o 0 K U % t - 9 ] ( Q V e w ' 7  T j F  ' 2 w F J  M N b 3 J E < ` B e H B : 0  s G + @ V 2 S    �  �  U  �  E  �    �  m    �  �  �    U  G  �    M  �  =  (  �  %  �  �  	  m  �  �  �  d  �  4  �  D  s  P  �  �  �  �  n    ]      �  �  �  f  �  W  �  �  h  8  $  �  �  �  �  b  J  �  >  +  �  o  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  %  )  .  3  3  2  1  -  '  !    
  �  �  �  �  �  �  �  �    t  �  b  �  �  )  �  �  �    )  3  2  �  �  �  �  �  3  �  �  �  �  �  �  �  �  �  g  C  '    �  �  `    �  �  M  �  �    '  -  (      �  �  �  Q  �  {    �  >  �  �   �  �  �  �  �  �  �  �  �  j  N  5       �  �  �  �  �  �  f      #    �  �  �  �  �  �  i  P  8  <  v  �  �  c  D  $  �  �    h  W  D  )    �  �  �  X  '  �  �  �  9  �  k   �  4  0  *  $      �  �  �  �  �  ]    �  �  �  �  V    �  �  �  �  �  �            �  �  �  U  �  ~  �    @  ^  ]  l  ~  �  �  �  �  �  r  E    �  �  �  q  H    �  �  �  )  H  p  �  �  �  �  �          �  �  �  Z  �  �  �   �  �  �  �  �  �  �  �  |  e  H    �  �  f    �  y  =    �  H  p  z    i  D  �  s  u  p  l  m  �  �  �  �  �  �  
�  �  "  <  O  Y  W  H  3    �  �  |  =  �    �  �  -  �  �  U  >  5  '    	  �  �  �  {  D    �  f    �  )  �  "  K  �  �  �  �  �  �  �  �  �  z  o  f  ^  U  I  =  3  *       
  �  �  �  �  �  y  h  T  @  ,  
  �  �    \  :     �   �   �  �  �  %  D  X  Z  J  1    �  �  m  "  �  �  >  1    �  
  �  	q  	�  	�  	�  	�  	�  	�  	�  	H  �  �    �    x  �  P  )  �  �  �  �  �  �  �  �  }  z  w  s  n  i  d  _  [  V  Q  L  G    �  �  �  �  �  �  �  �  �  �  ~  T  E  .     �  �  :   �    
     �   �   �   �   �   �   �   �   �   �   �   �   }   q   f   Z   N  �  �  �  �  �  �  �  �  �  �  �  �  �    x  q  j  c  \  U  	�  
�    �  �  �  �  �  �  �  �  �  z  (  
�  
  	`  �  �    ]  �  �  �  �  �  �  �  j  6  �  �  W     �  :  �  >  �     �  f  �  �    @  T  P  C  :  1  "    �  �  +  �  �  �  I  F  8  +        �  �  �  �  �  {  T  ,    $  /      �  �  �  �  �  �  �  �  �  r  S  4    �  �  �  `    �  z  (        .  e  |  �  �  �  �  �  �  �    n  [  B  )  �  �  �  �  s  a  I  3      �  �  �  c  7    �  �  �  [  `  T  �  �  �  x  g  Z  �  �  �  |  n  Q  -    �  �  X  �  �  $    )  +  %      �  �  �  l  3  �  �  �  Z    �  2  S     �  �  ~  t  g  X  I  6  "    �  �  �  y  V  :  $      �  0      �  �  �  �  �  �  r  V  7    �  �  ]    �  �  7  >  2  2  A  ,    �  �  �  �  n  O  )  #  \  J    �  �  �          �  �  �  �  �  �  o  K  %  �  �  �  �  u  \  C  �  �  �  �  �  |  b  B    �  �  �  b  .    �  �  ~  �  j  N  3  !  "  1  ?  L  T  P  C  0    �  �  �  P    �  W  �  h  N  D  B  I  >  5    �  �  �  N    �  U  �  z  �  �   �  i  Y  E  1  #    
  �  �  �  �  �  }  T  J  &    �  �  -  �  �  �  �  r  W  9    �  �  �  �  W  +  �  �  �  �  Y  .  �    ^  �  �    -  9  7  +    �  �  z    �  �      x  �  �  �  �  �  q  a  S  G  5    �  �  �  \  #  �  �  l  "  d  �  �  �  �  �  �  �  �  �  �  �  �  Z     �  �  )  �  �  �  �  }  u  m  a  V  K  >  /         �  �  �  �  �  �  k  '       �  �  �  b  &  �  �  1  �  |    �  X  �  �     �  F  c  m  j  T  (  (  H  ?  5    �  V  �  i  �  _  �    L  �  �  �  �  �  �  �  �  �  �  q  Y  <    �  �  �  �  �  �  �  �  �  x  k  ^  M  <  (      �  �  �  �  �  �  �  �  �  �        �  �  �  �  �  �  w  `  H  ,    �  �  �  �      h  Q  8    �  �  �  �  n  L  *    �  �  �  �  �  �  s  �           �  �  �  �  �  �  �  �  �  �  }  l  Z  9    +  >  H  A  4  %    �  �  �  f  (  �  �  �  j  "  �  )    l  h  e  a  ^  Z  V  R  K  ?  3  (    �  �  �  �  �  y  ^  >  4  *  !        �  �  �  �  �  �  �  g  G  '    �  �  �  �  �  �  �  �  �  �  �  c  .  �  �  }  <  �  �  .   �   �  U  N  G  @  9  1  $    	  �  �  �  �  �  �  �  (  S    �  _  Y  P  B  +    �  �  �  �  �  �  �    :  �  �  \   �   l          �  �  �  �  i  5  �  �    �  �  {  @    �  p  /  +  -    �  �  �  z  B  #  �  �  �  �  �  �  u  C  
  �  	h  	~  	o  	Z  	@  	  �  �  �  l  .  �  g  �  V  �  %  �  �      �  �  �  �  �  �  �  �  �  �  �    e  K  1       �  +    �  �  �  �  �  �  �  s  b  K  8  &      �  �  �  �  +  v  p  i  b  X  G  5  $          !  &  ,  1  ,  %      Y  N  >  (    �  �  �  �  z  V  /  	  �  �  �  J    �  c  �  �  �  �  �  v  `  I  1    �  �  �  J  �  �  8  �  �  9  D  J  O  U  [  `  c  e  g  h  j  l  o  z  �  �  �  �  �  �  �  �  	  	5  	*  	  �  �  �  O    �    +  �  k  �  �  �  �  �  �  �  {  c  J  0    �  �  �  8  �  �  E  �  �  B  �  w  E  ?  8  2  ,  %              +  9  G  U  b  p  ~  �